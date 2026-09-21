"""Strict parsing of the WN18/WN18RR entity-to-POS mapping artifact."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict


ENTITY_ID_RE = re.compile(r"^[0-9]{8}$")
INTELLIGIBLE_NAME_RE = re.compile(
    r"^__(?P<lemma>.+)_(?P<tag>NN|VB|JJ|RB)_(?P<sense>[0-9]+)$"
)
WORDNET_POS_BY_TAG: Dict[str, str] = {
    "NN": "n",
    "VB": "v",
    "JJ": "a",
    "RB": "r",
}


@dataclass(frozen=True)
class WordNetMappingRow:
    """One validated row from ``wordnet-mlj12-definitions.txt``."""

    entity_id: str
    offset: int
    intelligible_name: str
    lemma_text: str
    source_pos_tag: str
    wordnet_pos: str
    sense_index: int
    source_definition: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_definition_line(line: str) -> WordNetMappingRow:
    """Parse one mapping row without guessing or trying alternative POS values."""

    stripped = line.rstrip("\r\n")
    fields = stripped.split("\t")
    if len(fields) != 3:
        raise ValueError("WordNet mapping row must contain exactly three tab-separated fields")

    entity_id, intelligible_name, definition = fields
    if not ENTITY_ID_RE.fullmatch(entity_id):
        raise ValueError("WordNet entity ID must be exactly eight decimal digits")

    match = INTELLIGIBLE_NAME_RE.fullmatch(intelligible_name)
    if match is None:
        raise ValueError("Intelligible name must end in _NN_, _VB_, _JJ_, or _RB_ plus a sense index")

    definition = definition.strip()
    if not definition:
        raise ValueError("WordNet source definition must not be empty")

    lemma_source = match.group("lemma")
    lemma_text = lemma_source.replace("_", " ").strip()
    if not lemma_text:
        raise ValueError("WordNet lemma text must not be empty")

    source_pos_tag = match.group("tag")
    return WordNetMappingRow(
        entity_id=entity_id,
        offset=int(entity_id),
        intelligible_name=intelligible_name,
        lemma_text=lemma_text,
        source_pos_tag=source_pos_tag,
        wordnet_pos=WORDNET_POS_BY_TAG[source_pos_tag],
        sense_index=int(match.group("sense")),
        source_definition=definition,
    )
