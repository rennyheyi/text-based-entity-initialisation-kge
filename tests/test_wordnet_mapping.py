import unittest

from thesis.wordnet_mapping import parse_definition_line


class WordNetMappingTest(unittest.TestCase):
    def test_noun_mapping_is_parsed_without_pos_guessing(self):
        row = parse_definition_line(
            "02483564\t__symphalangus_NN_1\tused in some classifications for the siamangs\n"
        )
        self.assertEqual("02483564", row.entity_id)
        self.assertEqual(2483564, row.offset)
        self.assertEqual("NN", row.source_pos_tag)
        self.assertEqual("n", row.wordnet_pos)
        self.assertEqual("symphalangus", row.lemma_text)

    def test_all_declared_pos_tags_have_fixed_mappings(self):
        examples = {
            "NN": "n",
            "VB": "v",
            "JJ": "a",
            "RB": "r",
        }
        for tag, expected in examples.items():
            with self.subTest(tag=tag):
                row = parse_definition_line(
                    "00000001\t__example_{}_2\ta non-empty definition".format(tag)
                )
                self.assertEqual(expected, row.wordnet_pos)

    def test_invalid_identifier_fails(self):
        with self.assertRaises(ValueError):
            parse_definition_line("1740\t__entity_NN_1\ta definition")

    def test_missing_pos_fails_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            parse_definition_line("00001740\tentity\ta definition")

    def test_empty_definition_fails(self):
        with self.assertRaises(ValueError):
            parse_definition_line("00001740\t__entity_NN_1\t")


if __name__ == "__main__":
    unittest.main()
