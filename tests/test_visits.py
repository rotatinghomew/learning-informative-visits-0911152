import copy
import json
from pathlib import Path
import unittest

import numpy as np

from learning.codec import digest, encode
from learning.domain import ANALYSES, validate
from learning.visits import (
    ANALYSIS,
    DEFAULT_DELTAS,
    MIN_ANSWERED,
    parse_parameters,
    visit_adjusted_trend,
)


def release(name="term-a"):
    return json.loads(Path("data/releases.json").read_text(encoding="utf-8"))[name]


# 多数契约测试只检查结构与点估计，用较少的 bootstrap 次数控制测试时长；
# 确定性与顺序无关性测试仍使用默认 1000 次完整重抽。
FAST_PARAMS = {"bootstrap_replicates": 150}


class VisitTrendContractTests(unittest.TestCase):
    def setUp(self):
        self.tables = release()

    def test_result_is_serializable_and_deterministic(self):
        first = visit_adjusted_trend(self.tables, {})
        encode(first)
        second = visit_adjusted_trend(self.tables, {})
        self.assertEqual(digest(first), digest(second))

    def test_independent_of_row_order(self):
        before = visit_adjusted_trend(self.tables, {})
        reordered = {name: list(reversed(rows)) for name, rows in self.tables.items()}
        after = visit_adjusted_trend(reordered, {})
        self.assertEqual(digest(before), digest(after))

    def test_term_b_matches_term_a(self):
        other = release("term-b")
        self.assertEqual(digest(visit_adjusted_trend(self.tables, {})),
                         digest(visit_adjusted_trend(other, {})))

    def test_does_not_mutate_tables(self):
        before = digest(self.tables)
        visit_adjusted_trend(self.tables, FAST_PARAMS)
        self.assertEqual(before, digest(self.tables))

    def test_registered_in_domain_catalog(self):
        self.assertIn(ANALYSIS, ANALYSES)

    def test_groups_are_split_by_enrollment(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        self.assertEqual(set(result["groups"]), {"non_participants", "course_participants"})
        self.assertEqual(result["groups"]["non_participants"]["enrolled"], 0)
        self.assertEqual(result["groups"]["course_participants"]["enrolled"], 1)

    def test_periods_cover_horizon_in_bins(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        for body in result["groups"].values():
            self.assertEqual([p["bin"] for p in body["periods"]], [0, 1, 2, 3, 4, 5])
            self.assertEqual(body["periods"][0]["day_start"], 0)
            self.assertEqual(body["periods"][-1]["day_end"], 180)

    def test_reweighting_differs_from_raw_respondent_mean(self):
        # 调整后的估计基于应答倾向加权，不应退化为按回访记录求平均。
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        differences = [
            abs(p["adjusted_mean"] - p["raw_respondent_mean"])
            for body in result["groups"].values() for p in body["periods"]
            if p["adjusted_mean"] is not None
        ]
        self.assertTrue(any(value > 1e-6 for value in differences))

    def test_window_exit_bin_is_only_conditional(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        for body in result["groups"].values():
            last = body["periods"][-1]
            self.assertLess(last["eligible_people"], MIN_ANSWERED)
            self.assertEqual(last["evidence_status"], "conditional_only")
            self.assertIn(5, body["conditional_bins"])
            self.assertEqual(body["comparable_bins"], [0, 1, 2, 3, 4])

    def test_contact_outcomes_have_distinct_meaning(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        for body in result["groups"].values():
            counts = body["coverage"]
            self.assertGreater(counts["not_contacted_person_bins"], 0)
            self.assertGreater(counts["failed_contact_person_bins"], 0)
            self.assertGreater(counts["answered_person_bins"], 0)
            self.assertGreater(counts["out_of_window_contacts"], 0)
            # 三类 person-bin 之和等于全部合格 person-bin。
            self.assertEqual(
                counts["not_contacted_person_bins"]
                + counts["failed_contact_person_bins"]
                + counts["answered_person_bins"],
                counts["eligible_person_bins"],
            )

    def test_repeated_visits_count_once_per_person_bin(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        enrolled_by_person = {row["person"]: int(row["enrolled"]) for row in self.tables["people"]}
        for body in result["groups"].values():
            answered = sum(p["answered_people"] for p in body["periods"])
            group_visits = sum(1 for row in self.tables["visits"]
                               if enrolled_by_person[row["person"]] == body["enrolled"])
            # person-bin 应答单元数不超过该组回访记录数（同人同桶多次回访只算一格）。
            self.assertLessEqual(answered, group_visits)

    def test_sensitivity_brackets_and_deltas_present(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        for body in result["groups"].values():
            for period in body["periods"]:
                keys = {float(key) for key in period["delta_adjusted_mean"]}
                self.assertEqual(keys, set(DEFAULT_DELTAS))
                values = [value for value in period["delta_adjusted_mean"].values()
                          if value is not None]
                if values:
                    # delta 越大，插补的未应答者体验越高，桶均值单调不减。
                    ordered = [period["delta_adjusted_mean"][str(delta)]
                               for delta in sorted(DEFAULT_DELTAS)]
                    self.assertEqual(ordered, sorted(ordered))

    def test_result_refuses_causal_interpretation(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        self.assertIn("不能解释为课程造成的改善", result["interpretation"]["causal_claim"])

    def test_bootstrap_confidence_intervals(self):
        result = visit_adjusted_trend(self.tables, FAST_PARAMS)
        for body in result["groups"].values():
            for period in body["periods"][:5]:
                low, high = period["confidence_interval_95"]
                self.assertLess(low, high)
                self.assertLess(low, period["adjusted_mean"])
                self.assertLess(period["adjusted_mean"], high)

    def test_limited_release_returns_nulls_not_estimates(self):
        result = visit_adjusted_trend(release("limited"), {})
        self.assertEqual(result["diagnostics"]["reason"], "no_visits_in_release")
        for body in result["groups"].values():
            self.assertEqual(body["contact_propensity_model"], "not_estimated")
            self.assertEqual(body["response_propensity_model"], "not_estimated")
            self.assertEqual(body["insufficient_bins"], list(range(6)))
            for period in body["periods"]:
                self.assertIsNone(period["adjusted_mean"])
                self.assertIsNone(period["raw_respondent_mean"])
                self.assertIsNone(period["confidence_interval_95"])
                self.assertEqual(period["evidence_status"], "insufficient_evidence")
            self.assertGreater(body["coverage"]["out_of_window_contacts"], 0)
            self.assertEqual(
                body["coverage"]["not_contacted_person_bins"]
                + body["coverage"]["failed_contact_person_bins"],
                body["coverage"]["eligible_person_bins"],
            )


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.tables = release()

    def test_defaults_and_custom_grid(self):
        settings = parse_parameters({})
        self.assertEqual(settings["bin_days"], 30)
        self.assertEqual(settings["horizon_days"], 180)
        custom = visit_adjusted_trend(
            self.tables, {"bin_days": 15, "deltas": [-5, 0, 5], "bootstrap_replicates": 150})
        self.assertEqual(len(custom["groups"]["non_participants"]["periods"]), 12)
        for body in custom["groups"].values():
            for period in body["periods"]:
                self.assertEqual(set(period["delta_adjusted_mean"]), {"-5.0", "0.0", "5.0"})

    def test_invalid_parameters_rejected(self):
        bad_cases = [
            {"bin_days": 0}, {"bin_days": 181}, {"horizon_days": 100},
            {"bin_days": 7, "horizon_days": 100}, {"bootstrap_replicates": 10},
            {"bootstrap_replicates": 50000}, {"deltas": []}, {"deltas": [200]},
            {"deltas": "x"}, {"seed": "x"}, {"bin_days": True},
        ]
        for parameters in bad_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    visit_adjusted_trend(self.tables, parameters)

    def test_seed_changes_bootstrap_but_not_point_estimates(self):
        first = visit_adjusted_trend(self.tables, {"seed": 1, "bootstrap_replicates": 150})
        second = visit_adjusted_trend(self.tables, {"seed": 2, "bootstrap_replicates": 150})
        for group in first["groups"]:
            for p1, p2 in zip(first["groups"][group]["periods"],
                              second["groups"][group]["periods"]):
                self.assertEqual(p1["adjusted_mean"], p2["adjusted_mean"])


class SelectionMechanismTests(unittest.TestCase):
    def setUp(self):
        self.tables = release()

    def test_ipw_reweights_away_from_respondent_selection(self):
        # 构造极端选择：删除一半低分行的回访，调整估计应高于应答者原始均值。
        tables = copy.deepcopy(self.tables)
        kept = [row for row in tables["visits"] if row["wellbeing"] >= 55]
        removed = [row for row in tables["visits"] if row["wellbeing"] < 55][:60]
        tables["visits"] = kept + removed
        validate(tables)
        result = visit_adjusted_trend(tables, {"bootstrap_replicates": 200})
        shifts = [
            p["adjusted_mean"] - p["raw_respondent_mean"]
            for body in result["groups"].values() for p in body["periods"]
            if p["adjusted_mean"] is not None and p["answered_people"] >= MIN_ANSWERED
        ]
        self.assertTrue(any(abs(value) > 1e-6 for value in shifts))

    def test_synthetic_mar_recovers_group_mean(self):
        # 合成数据：体验与应答都只依赖已观测协变量，IPW 应接近无偏。
        rng = np.random.default_rng(7)
        people = [{"person": f"p{i:03d}", "age": 65 + i % 10, "site": "s0",
                   "enrolled": int(i >= 48)} for i in range(96)]
        windows = [{"person": p["person"], "start": 0, "end": 180, "reason": "观察结束"}
                   for p in people]
        contacts, visits = [], []
        truth = {p["person"]: 55 + 8 * (p["enrolled"] - 0.5) for p in people}
        for person in people:
            base = truth[person["person"]]
            for day in range(10, 175, 20):
                contacts.append({"person": person["person"], "day": day,
                                 "contacted": True, "scheduled": day % 40 == 10})
                # 应答概率随“真实体验”变化，但体验是固定组均值加噪声，
                # 协变量（年龄、组别、桶）可捕捉该选择，满足 MAR 设定。
                probability = 1 / (1 + np.exp(-(base - 50) / 20))
                if rng.random() < probability:
                    visits.append({"person": person["person"], "day": day,
                                   "wellbeing": round(base + rng.normal(0, 3), 2),
                                   "mobility": 3.0})
        tables = {"people": people, "contacts": contacts, "visits": visits, "windows": windows}
        validate(tables)
        result = visit_adjusted_trend(tables, {"bootstrap_replicates": 300})
        for group, expected in ((0, 51.0), (1, 59.0)):
            body = result["groups"]["non_participants" if group == 0 else "course_participants"]
            comparable = [p for p in body["periods"] if p["evidence_status"] == "comparable"]
            self.assertGreaterEqual(len(comparable), 4)
            for period in comparable:
                self.assertAlmostEqual(period["adjusted_mean"], expected, delta=4.0)

    def test_synthetic_two_stage_selection_is_corrected(self):
        # 同时存在“是否安排联系”和“是否取得应答”两层选择，且都只依赖
        # 已观测协变量（MAR）。两阶段 IPW 应恢复窗内合格总体的真实均值，
        # 而只看应答者的原始均值会因选择而偏高。
        rng = np.random.default_rng(11)
        n_people = 120
        people = [{"person": f"p{i:03d}", "age": 60 + (i % 20), "site": "s0",
                   "enrolled": int(i >= n_people // 2)} for i in range(n_people)]
        windows = [{"person": p["person"], "start": 0, "end": 180, "reason": "观察结束"}
                   for p in people]
        contacts, visits = [], []
        truth = {p["person"]: 50.0 + 10.0 * p["enrolled"] for p in people}
        for person in people:
            base = truth[person["person"]]
            age_factor = (int(person["age"]) - 70) / 20
            for day in range(15, 175, 30):
                # 第一阶段：被联系概率随已观测的年龄与课程组变化。
                contact_p = 1 / (1 + np.exp(-(0.4 + 0.6 * person["enrolled"] + age_factor)))
                if rng.random() >= contact_p:
                    continue
                contacts.append({"person": person["person"], "day": day,
                                 "contacted": True, "scheduled": day % 60 == 15})
                # 第二阶段：被联系者的应答概率同样只依赖已观测特征。
                response_p = 1 / (1 + np.exp(-(0.2 + 0.4 * person["enrolled"] - age_factor)))
                if rng.random() < response_p:
                    visits.append({"person": person["person"], "day": day,
                                   "wellbeing": round(base + rng.normal(0, 2), 2),
                                   "mobility": 3.0})
        tables = {"people": people, "contacts": contacts, "visits": visits, "windows": windows}
        validate(tables)
        result = visit_adjusted_trend(tables, {"bootstrap_replicates": 300})
        for group, expected in ((0, 50.0), (1, 60.0)):
            body = result["groups"]["non_participants" if group == 0 else "course_participants"]
            self.assertEqual(body["contact_propensity_model"], "logistic_l2")
            self.assertEqual(body["response_propensity_model"], "logistic_l2")
            self.assertGreater(body["coverage"]["not_contacted_person_bins"], 0)
            self.assertGreater(body["coverage"]["failed_contact_person_bins"], 0)
            comparable = [p for p in body["periods"] if p["evidence_status"] == "comparable"]
            self.assertGreaterEqual(len(comparable), 3)
            for period in comparable:
                self.assertAlmostEqual(period["adjusted_mean"], expected, delta=5.0)


if __name__ == "__main__":
    unittest.main()
