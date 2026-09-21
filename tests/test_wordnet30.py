import unittest

from thesis.wordnet30 import (
    WordNet30Error,
    definition_from_gloss,
    mapping_gloss_alignment,
    parse_data_file,
    parse_sense_index,
    normalise_wordnet_lemma,
    verify_mapping_row,
)
from thesis.wordnet_mapping import parse_definition_line


class WordNet30Test(unittest.TestCase):
    def setUp(self):
        self.entries = parse_data_file(
            b"00001740 03 n 01 entity 00 000 | that which exists\n",
            lookup_pos="n",
        )
        self.senses = parse_sense_index(
            b"entity%1:03:00:: 00001740 1 0\n"
        )

    def test_exact_offset_pos_lemma_sense_and_definition_resolve(self):
        row = parse_definition_line(
            "00001740\t__entity_NN_1\tthat which exists\n"
        )
        entry = verify_mapping_row(row, self.entries, self.senses)
        self.assertEqual(("entity",), entry.lemmas)
        self.assertEqual("that which exists", entry.definition)
        self.assertEqual(
            "exact_wordnet_gloss", mapping_gloss_alignment(row, entry)
        )

    def test_target_gloss_may_be_identified_inside_contaminated_mapping_field(self):
        row = parse_definition_line(
            "00001740\t__entity_NN_1\t"
            "an unrelated noun gloss that which exists an unrelated verb gloss\n"
        )
        entry = verify_mapping_row(row, self.entries, self.senses)
        self.assertEqual(
            "target_gloss_with_extra_text",
            mapping_gloss_alignment(row, entry),
        )

    def test_wrong_pos_does_not_fall_back(self):
        row = parse_definition_line(
            "00001740\t__entity_VB_1\tthat which exists\n"
        )
        with self.assertRaises(WordNet30Error):
            verify_mapping_row(row, self.entries, self.senses)

    def test_definition_and_sense_mismatches_fail(self):
        wrong_definition = parse_definition_line(
            "00001740\t__entity_NN_1\ta different definition\n"
        )
        with self.assertRaisesRegex(WordNet30Error, "entity 00001740"):
            verify_mapping_row(wrong_definition, self.entries, self.senses)

        wrong_sense = parse_definition_line(
            "00001740\t__entity_NN_2\tthat which exists\n"
        )
        with self.assertRaises(WordNet30Error):
            verify_mapping_row(wrong_sense, self.entries, self.senses)

    def test_adjective_satellite_uses_adjective_lookup_pos(self):
        entries = parse_data_file(
            b"00000001 00 s 01 satellite 00 000 | a satellite adjective\n",
            lookup_pos="a",
        )
        senses = parse_sense_index(
            b"satellite%5:00:00:: 00000001 2 0\n"
        )
        row = parse_definition_line(
            "00000001\t__satellite_JJ_2\ta satellite adjective\n"
        )
        self.assertEqual("s", verify_mapping_row(row, entries, senses).synset_type)

    def test_adjective_syntax_marker_is_not_part_of_lexical_lemma(self):
        entries = parse_data_file(
            b"00000002 00 a 01 awake(p) 00 000 | not asleep\n",
            lookup_pos="a",
        )
        senses = parse_sense_index(b"awake%3:00:00:: 00000002 1 0\n")
        row = parse_definition_line("00000002\t__awake_JJ_1\tnot asleep\n")
        self.assertEqual("awake(p)", verify_mapping_row(row, entries, senses).lemmas[0])
        self.assertEqual(
            "awake", normalise_wordnet_lemma("awake(p)", lookup_pos="a")
        )
        self.assertEqual(
            "example(p)", normalise_wordnet_lemma("example(p)", lookup_pos="n")
        )

    def test_gloss_examples_are_not_part_of_definition(self):
        self.assertEqual(
            "the definition",
            definition_from_gloss(' the definition; "example one"; "example two" '),
        )


if __name__ == "__main__":
    unittest.main()
