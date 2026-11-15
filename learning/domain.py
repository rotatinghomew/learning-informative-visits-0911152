from collections import Counter, defaultdict
import math

import numpy as np

from learning.codec import numeric, require_fields, unique
from learning.visits import ANALYSIS as VISIT_TREND_ANALYSIS, visit_adjusted_trend

PRODUCT = '学员回访分析'
KEYS = {'people': ['person'], 'contacts': ['person', 'day'], 'visits': ['person', 'day'], 'windows': ['person']}


def validate(tables):
    if set(tables) != set(KEYS):
        raise ValueError("资料表集合与产品契约不一致")
    for name, fields in KEYS.items():
        require_fields(tables[name], fields)
        unique(tables[name], fields)
    people = {row["person"] for row in tables["people"]}
    contacts = {(row["person"], row["day"]) for row in tables["contacts"]}
    for row in tables["people"]:
        if not isinstance(row["enrolled"], int) or isinstance(row["enrolled"], bool) or row["enrolled"] not in (0, 1):
            raise ValueError("课程参与标记必须是 0 或 1")
    for row in tables["contacts"]:
        if not isinstance(row["contacted"], bool) or not isinstance(row["scheduled"], bool):
            raise ValueError("联系状态标记必须是布尔值")
    for row in tables["windows"]:
        if row["person"] not in people or row["start"] >= row["end"]:
            raise ValueError("观察窗口不合法")
        numeric(row["start"], "窗口起点")
        numeric(row["end"], "窗口终点")
    for row in tables["visits"]:
        if (row["person"], row["day"]) not in contacts:
            raise ValueError("回访没有联系日志")
        numeric(row["wellbeing"], "生活体验")
        numeric(row["mobility"], "出行评分")


def overview(tables, parameters):
    observed = defaultdict(list)
    for row in tables["visits"]:
        observed[row["person"]].append(row["wellbeing"])
    return {"people": [{"person": row["person"], "responses": len(observed[row["person"]]),
                        "raw_mean": float(np.mean(sorted(observed[row["person"]]))) if observed[row["person"]] else None}
                       for row in sorted(tables["people"], key=lambda row: row["person"])], "contacts": len(tables["contacts"])}


ANALYSES = {"overview": overview, VISIT_TREND_ANALYSIS: visit_adjusted_trend}
