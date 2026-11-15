from collections import Counter
import json
from pathlib import Path
import unittest

from learning.domain import overview, validate


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads(Path("data/releases.json").read_text(encoding="utf-8"))["term-a"]

    def test_raw_person_means_reconcile(self):
        result = overview(self.tables, {})
        self.assertEqual(sum(row["responses"] for row in result["people"]), len(self.tables["visits"]))
        for summary in result["people"]:
            values = [row["wellbeing"] for row in self.tables["visits"] if row["person"] == summary["person"]]
            self.assertAlmostEqual(summary["raw_mean"], sum(values)/len(values))

    def test_contacts_without_responses_are_not_measurements(self):
        self.tables["visits"] = []
        result = overview(self.tables, {})
        self.assertGreater(result["contacts"], 0)
        self.assertTrue(all(row["raw_mean"] is None for row in result["people"]))

    def test_response_without_contact_rejected(self):
        self.tables["visits"][0]["day"] = 1000
        with self.assertRaises(ValueError):
            validate(self.tables)
