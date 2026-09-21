"""Minimal, auditable parser for the pinned Princeton WordNet 3.0 files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple

from .wordnet_mapping import WordNetMappingRow


DATA_LINE_RE = re.compile(r"^[0-9]{8}\s")
ADJECTIVE_SYNTAX_MARKER_RE = re.compile(r"\((?:a|p|ip)\)$")
SENSE_POS = {"1": "n", "2": "v", "3": "a", "4": "r", "5": "a"}


class WordNet30Error(RuntimeError):
    """Raised when the pinned lexical source or its alignment is invalid."""


@dataclass(frozen=True)
class WordNetEntry:
    offset: int
    lookup_pos: str
    synset_type: str
    lemmas: Tuple[str, ...]
    gloss: str
    definition: str


def normalise_text(value: str) -> str:
    return " ".join(value.split())


def normalise_lemma(value: str) -> str:
    return normalise_text(value.replace("_", " ")).casefold()


def normalise_wordnet_lemma(value: str, *, lookup_pos: str) -> str:
    """Normalise a data-file lemma using WordNet's adjective marker rule."""

    if lookup_pos not in {"n", "v", "a", "r"}:
        raise WordNet30Error("Unsupported WordNet lookup POS")
    if lookup_pos == "a":
        value = ADJECTIVE_SYNTAX_MARKER_RE.sub("", value)
    return normalise_lemma(value)


def definition_from_gloss(gloss: str) -> str:
    definition = re.split(r';\s+"', gloss, maxsplit=1)[0]
    definition = normalise_text(definition)
    if not definition:
        raise WordNet30Error("WordNet synset has an empty definition")
    return definition


def parse_data_file(raw: bytes, *, lookup_pos: str) -> Dict[Tuple[int, str], WordNetEntry]:
    if lookup_pos not in {"n", "v", "a", "r"}:
        raise WordNet30Error("Unsupported WordNet lookup POS")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WordNet30Error("WordNet data file is not UTF-8") from error
    expected_types = {"a", "s"} if lookup_pos == "a" else {lookup_pos}
    entries: Dict[Tuple[int, str], WordNetEntry] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not DATA_LINE_RE.match(line):
            continue
        left, separator, gloss = line.partition("|")
        if not separator:
            raise WordNet30Error("WordNet data line {} has no gloss separator".format(line_number))
        fields = left.split()
        if len(fields) < 4:
            raise WordNet30Error("WordNet data line {} is truncated".format(line_number))
        try:
            offset = int(fields[0])
            synset_type = fields[2]
            word_count = int(fields[3], 16)
        except ValueError as error:
            raise WordNet30Error("WordNet data header is malformed") from error
        if synset_type not in expected_types:
            raise WordNet30Error("WordNet synset type differs from its data file")
        lemma_end = 4 + 2 * word_count
        if word_count <= 0 or len(fields) < lemma_end:
            raise WordNet30Error("WordNet lemma block is malformed")
        lemmas = tuple(fields[4 + 2 * index] for index in range(word_count))
        if any(not normalise_lemma(lemma) for lemma in lemmas):
            raise WordNet30Error("WordNet synset has an empty lemma")
        key = (offset, lookup_pos)
        if key in entries:
            raise WordNet30Error("Duplicate WordNet offset/POS entry")
        entries[key] = WordNetEntry(
            offset=offset,
            lookup_pos=lookup_pos,
            synset_type=synset_type,
            lemmas=lemmas,
            gloss=normalise_text(gloss),
            definition=definition_from_gloss(gloss),
        )
    if not entries:
        raise WordNet30Error("WordNet data file contains no synsets")
    return entries


def parse_sense_index(raw: bytes) -> Dict[Tuple[int, str, str], Set[int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WordNet30Error("WordNet sense index is not UTF-8") from error
    result: Dict[Tuple[int, str, str], Set[int]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4 or "%" not in fields[0]:
            raise WordNet30Error("Malformed index.sense line {}".format(line_number))
        sense_key, offset_text, sense_number_text, _ = fields
        lemma, sense_metadata = sense_key.split("%", 1)
        pos_code = sense_metadata[:1]
        if pos_code not in SENSE_POS:
            raise WordNet30Error("Unknown WordNet sense-key POS code")
        try:
            offset = int(offset_text)
            sense_number = int(sense_number_text)
        except ValueError as error:
            raise WordNet30Error("Malformed WordNet sense index number") from error
        if sense_number <= 0:
            raise WordNet30Error("WordNet sense number must be positive")
        key = (offset, SENSE_POS[pos_code], normalise_lemma(lemma))
        result.setdefault(key, set()).add(sense_number)
    if not result:
        raise WordNet30Error("WordNet sense index is empty")
    return result


def verify_mapping_row(
    row: WordNetMappingRow,
    entries: Mapping[Tuple[int, str], WordNetEntry],
    senses: Mapping[Tuple[int, str, str], Set[int]],
) -> WordNetEntry:
    key = (row.offset, row.wordnet_pos)
    if key not in entries:
        raise WordNet30Error(
            "No WordNet entry for entity {} with POS {}".format(
                row.entity_id, row.wordnet_pos
            )
        )
    entry = entries[key]
    mapped_lemma = normalise_lemma(row.lemma_text)
    lexical_lemmas = {
        normalise_wordnet_lemma(lemma, lookup_pos=entry.lookup_pos)
        for lemma in entry.lemmas
    }
    if mapped_lemma not in lexical_lemmas:
        raise WordNet30Error("Mapped lemma is absent from the resolved WordNet synset")
    mapping_gloss_alignment(row, entry)
    sense_numbers = senses.get((row.offset, row.wordnet_pos, mapped_lemma), set())
    if row.sense_index not in sense_numbers:
        raise WordNet30Error("Mapped sense index differs from WordNet index.sense")
    return entry


def mapping_gloss_alignment(
    row: WordNetMappingRow,
    entry: WordNetEntry,
) -> str:
    """Classify the mapping gloss without using it as entity evidence.

    The KG-BERT mapping may concatenate glosses for equal numeric offsets from
    different POS files. The selected WordNet 3.0 gloss must still occur
    verbatim after whitespace normalisation; otherwise the mapping is
    incompatible and the audit fails.
    """

    mapped_gloss = normalise_text(row.source_definition)
    if mapped_gloss == entry.gloss:
        return "exact_wordnet_gloss"
    if entry.gloss in mapped_gloss:
        return "target_gloss_with_extra_text"
    else:
        raise WordNet30Error(
            "Resolved WordNet gloss is absent for entity {} (POS {}): "
            "mapping={!r}; wordnet={!r}".format(
                row.entity_id,
                row.source_pos_tag,
                mapped_gloss,
                entry.gloss,
            )
        )
