"""visit_adjusted_trend：处理选择性联系与选择性应答的群体体验走势。

估计目标（estimand）
--------------------
在每个时间桶 b、课程参与组 g 内，估计当时仍处于个人观察窗口内的学员
群体的平均生活体验 E[wellbeing | group=g, bin=b, still in window]。
这是描述性走势，不是课程效应：enrolled 是登记组别而非随机分配，
调整只处理“谁被联系、谁回答”，不处理组间的基线差异。

选择过程分两层，含义不同，用两阶段逆概率加权分别建模：
1. 没有任何联系日志的 person-bin：站点当时没有安排联系。第一阶段估计
   “被联系”的概率 π_contact，把未安排联系视为窗内合格总体上的失访；
2. 有联系但没有本人测量的 person-bin：联系失败（尝试过但未取得回答）。
   第二阶段在被联系者中估计“取得本人测量”的概率 π_response。
观察单元的总权重为 1/(π_contact·π_response)，并按时间桶的边缘概率稳定化。
窗口外的联系只作历史资料，不进入估计（离开服务范围与窗内失访不同）。

识别假设与敏感性
----------------
主分析要求：
- 条件可交换性（MAR）：在观察到的协变量（既往测量、联系节奏、是否预定、
  年龄、时间桶）条件下，被联系与应答都与当时未观测的体验独立；
- 正值性：各协变量层内被联系与应答概率都显著大于 0。
回访时机既可能取决于已记录状态，也可能取决于未观测状态；后者无法用
现有资料检验，因此同时给出 delta 调整的边界分析（MNAR 情景）量化敏感
程度，并让每个时间桶带上证据状态（可比较 / 仅条件性结论 / 证据不足）。

同一个人的多次回访在面板里属于同一聚类：权重在 person-bin 层面计算，
推断使用按人重抽的分层 bootstrap，不把密集回访当作独立学员。
"""
from collections import defaultdict

import numpy as np

from learning.codec import numeric

ANALYSIS = "visit_adjusted_trend"
DEFAULT_BIN_DAYS = 30
DEFAULT_HORIZON_DAYS = 180
DEFAULT_DELTAS = (-10.0, 0.0, 10.0)
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 20261115
# 证据门槛：低于该应答人数或有效样本量的时间桶只能给条件性结论。
MIN_ANSWERED = 20
MIN_ESS_RATIO = 0.5
MIN_PROPENSITY = 0.1
MAX_WEIGHT_RATIO = 10.0
# 倾向模型的最小样本量与每层最少事件数，不足时退化为边缘概率并标注。
MIN_MODEL_ROWS = 30
MIN_MODEL_EVENTS = 5


def _finite_number(value, name):
    return numeric(value, name)


def parse_parameters(parameters):
    """校验并补全分析参数，参数非法时抛 ValueError。"""
    parameters = parameters or {}
    if not isinstance(parameters, dict):
        raise ValueError("分析参数不是对象")
    bin_days = parameters.get("bin_days", DEFAULT_BIN_DAYS)
    horizon_days = parameters.get("horizon_days", DEFAULT_HORIZON_DAYS)
    bootstrap = parameters.get("bootstrap_replicates", DEFAULT_BOOTSTRAP)
    seed = parameters.get("seed", DEFAULT_SEED)
    deltas = parameters.get("deltas", list(DEFAULT_DELTAS))
    if isinstance(bin_days, bool) or not isinstance(bin_days, int) or not 1 <= bin_days <= 180:
        raise ValueError("bin_days 必须是 1 到 180 之间的整数")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or not bin_days <= horizon_days <= 3650:
        raise ValueError("horizon_days 必须不小于 bin_days 且不超过 3650")
    if horizon_days % bin_days:
        raise ValueError("horizon_days 必须是 bin_days 的整数倍")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or not 100 <= bootstrap <= 20000:
        raise ValueError("bootstrap_replicates 必须是 100 到 20000 之间的整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed 必须是整数")
    if not isinstance(deltas, list) or not 1 <= len(deltas) <= 9:
        raise ValueError("deltas 必须是 1 到 9 个数值组成的列表")
    clean_deltas = []
    for value in deltas:
        clean_deltas.append(_finite_number(value, "delta"))
    if any(abs(value) > 100 for value in clean_deltas):
        raise ValueError("delta 的绝对值不能超过 100")
    return {
        "bin_days": bin_days,
        "horizon_days": horizon_days,
        "bootstrap_replicates": bootstrap,
        "seed": seed,
        "deltas": clean_deltas,
    }


def _build_panel(tables, settings):
    """构造 person × 时间桶面板，并登记每格的联系/应答/协变量信息。"""
    bin_days = settings["bin_days"]
    horizon = settings["horizon_days"]
    n_bins = horizon // bin_days
    attr = {row["person"]: row for row in tables["people"]}
    windows = {row["person"]: row for row in tables["windows"]}
    contacts_by_person = defaultdict(list)
    for row in tables["contacts"]:
        contacts_by_person[row["person"]].append(row)
    answers = {(row["person"], row["day"]): row for row in tables["visits"]}

    panel = []
    for person in sorted(attr):
        window = windows[person]
        start, end = window["start"], window["end"]
        person_contacts = sorted(contacts_by_person[person], key=lambda row: row["day"])
        history_values = []
        for bin_index in range(n_bins):
            bin_start = bin_index * bin_days
            bin_end = bin_start + bin_days
            eligible = start <= bin_start and end >= bin_end
            if not eligible:
                continue
            in_bin = [row for row in person_contacts if bin_start <= row["day"] < bin_end]
            prior = [row for row in person_contacts if row["day"] < bin_start]
            answered_rows = [answers[(person, row["day"])] for row in in_bin
                             if (person, row["day"]) in answers]
            outcome = float(np.mean([row["wellbeing"] for row in answered_rows])) if answered_rows else None
            mobility = float(np.mean([row["mobility"] for row in answered_rows])) if answered_rows else None
            panel.append({
                "person": person,
                "group": int(attr[person]["enrolled"]),
                "bin": bin_index,
                "bin_start": bin_start,
                "bin_end": bin_end,
                "contact_status": "answered" if answered_rows else ("failed" if in_bin else "not_contacted"),
                "n_contacts": len(in_bin),
                "n_scheduled": sum(1 for row in in_bin if row["scheduled"]),
                "n_prior_contacts": len(prior),
                "prior_mean": float(np.mean(history_values)) if history_values else None,
                "last_value": history_values[-1] if history_values else None,
                "has_prior": bool(history_values),
                "age": float(attr[person]["age"]),
                "outcome": outcome,
                "mobility": mobility,
            })
            history_values.extend(row["wellbeing"] for row in answered_rows)
    return panel, n_bins


def _design_matrix(ages, prior_mean, prior_missing, n_contacts, n_scheduled,
                   n_prior_contacts, bin_index, n_bins, reference=None):
    """组装倾向模型的设计矩阵（连续变量按全组合格面板统一中心化和缩放）。

    reference 为 None 时按当前数据计算 (center, scale) 并一并返回，保证联系
    阶段与应答阶段、点估计与 bootstrap 使用同一套标准化基准。
    """
    columns = ["intercept"]
    matrix = [np.ones(len(ages))]
    continuous = (("age", ages), ("prior_mean", prior_mean),
                  ("prior_missing", prior_missing.astype(float)),
                  ("n_contacts", n_contacts), ("n_scheduled", n_scheduled),
                  ("n_prior_contacts", n_prior_contacts))
    computed = {}
    for position, (name, values) in enumerate(continuous):
        if reference is None:
            center, scale = float(values.mean()), float(values.std())
            computed[position] = (center, scale)
        else:
            center, scale = reference[position]
        matrix.append(values if scale == 0 else (values - center) / scale)
        columns.append(name)
    for bin_id in range(1, n_bins):
        columns.append(f"bin_{bin_id}")
        matrix.append((bin_index == bin_id).astype(float))
    design = np.column_stack(matrix)
    if reference is None:
        return design, columns, computed
    return design, columns


def _design_rows(rows, reference=None, prior_fill_value=None):
    """兼容字典行的设计矩阵包装。

    reference 提供全组 (center, scale)，prior_fill_value 提供全组缺失既往
    测量的统一填充值，保证子集（被联系者）模型与全局面板基准一致。
    """
    ages = np.array([row["age"] for row in rows], dtype=float)
    prior = np.array([row["prior_mean"] if row["prior_mean"] is not None else np.nan for row in rows])
    prior_missing = np.isnan(prior)
    if prior_fill_value is None:
        prior_fill_value = float(np.nanmean(prior)) if (~prior_missing).any() else 0.0
    prior_filled = np.where(prior_missing, prior_fill_value, prior)
    n_bins = max(row["bin"] for row in rows) + 1
    result = _design_matrix(
        ages, prior_filled, prior_missing,
        np.array([row["n_contacts"] for row in rows], dtype=float),
        np.array([row["n_scheduled"] for row in rows], dtype=float),
        np.array([row["n_prior_contacts"] for row in rows], dtype=float),
        np.array([row["bin"] for row in rows]), n_bins, reference,
    )
    if reference is None:
        return result[0], result[2]
    return result[0], result[1]


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _stage_probabilities(rows, indicator, scales=None, prior_fill_value=None):
    """拟合某个选择阶段（被联系 / 取得应答）的岭罚 logistic 倾向。

    样本或事件数不足时返回 (None, 边缘概率)，调用方退化为边缘加权并标注。
    indicator 为与 rows 等长的 0/1 序列；scales 复用全组合格面板的标准化基准。
    """
    target = np.asarray(indicator, dtype=float)
    marginal = float(target.mean()) if len(rows) else 1.0
    if (len(rows) < MIN_MODEL_ROWS or target.sum() < MIN_MODEL_EVENTS
            or (1 - target).sum() < MIN_MODEL_EVENTS):
        return None, marginal
    design, _ = _design_rows(rows, scales, prior_fill_value)
    penalty = np.array([0.0] + [1.0] * (design.shape[1] - 1))
    return _irls_probabilities(design, target, penalty), marginal


def _irls_probabilities(design, target, penalty):
    """IRLS（牛顿法）求解岭罚 logistic 回归并返回预测概率。

    岭罚保证 Hessian 正定，牛顿方向天然是下降方向，因此用牛顿减量
    （梯度范数）判断收敛，省去每轮重复的完整损失计算。
    """
    beta = np.zeros(design.shape[1])
    for _ in range(25):
        probability = _sigmoid(design @ beta)
        gradient = design.T @ (probability - target) + penalty * beta
        if np.max(np.abs(gradient)) < 1e-8:
            break
        weight = probability * (1 - probability)
        hessian = design.T @ (weight[:, None] * design) + np.diag(penalty)
        beta -= np.linalg.solve(hessian, gradient)
    return _sigmoid(design @ beta)


def _two_stage_weights(group_rows, n_bins, arrays=None):
    """在一个课程组上拟合并计算两阶段稳定化权重与分桶诊断。

    arrays 为该组的列式表示，其 scales 保证点估计与 bootstrap 的标准化基准一致。
    返回每个被应答 person-bin 的 (bin, outcome, 总权重) 列表、逐桶诊断、
    两阶段模型名称，以及每个合格 person-bin 的被联系概率（供诊断）。
    """
    if arrays is None:
        arrays = _GroupArrays(group_rows, n_bins)
    contacted_flag = np.array([row["contact_status"] != "not_contacted" for row in group_rows], dtype=float)
    contact_prob, contact_marginal = _stage_probabilities(
        group_rows, contacted_flag, arrays.scales, arrays.prior_fill_value)
    if contact_prob is None:
        contact_prob = np.full(len(group_rows), contact_marginal)
        contact_model = "marginal_fallback"
    else:
        contact_model = "logistic_l2"

    contacted_rows = [row for row in group_rows if row["contact_status"] != "not_contacted"]
    response_flag = [1 if row["outcome"] is not None else 0 for row in contacted_rows]
    response_prob, response_marginal = _stage_probabilities(
        contacted_rows, response_flag, arrays.scales, arrays.prior_fill_value)
    if response_prob is None:
        response_prob = np.full(len(contacted_rows), response_marginal)
        response_model = "marginal_fallback"
    else:
        response_model = "logistic_l2"

    contact_lookup = {id(row): probability for row, probability in zip(group_rows, contact_prob)}
    bin_stats = defaultdict(lambda: {"eligible": 0, "contacted": 0, "answered": 0})
    for row in group_rows:
        stats = bin_stats[row["bin"]]
        stats["eligible"] += 1
        if row["contact_status"] != "not_contacted":
            stats["contacted"] += 1
        if row["outcome"] is not None:
            stats["answered"] += 1

    weighted_rows, diagnostics = [], {}
    for bin_index, stats in bin_stats.items():
        diagnostics[bin_index] = {
            "n_eligible": stats["eligible"], "n_contacted": stats["contacted"],
            "n_answered": stats["answered"], "ess": 0.0,
            "min_contact_propensity": None, "min_response_propensity": None,
            "max_weight": None,
        }
    for row, p_response in zip(contacted_rows, response_prob):
        if row["outcome"] is None:
            continue
        bin_index = row["bin"]
        stats = bin_stats[bin_index]
        p_contact = contact_lookup[id(row)]
        stable_contact = (stats["contacted"] / stats["eligible"]) / p_contact
        stable_response = (stats["answered"] / stats["contacted"]) / p_response
        weight = stable_contact * stable_response
        weighted_rows.append((bin_index, row["outcome"], float(weight),
                              float(p_contact), float(p_response)))
        diagnostic = diagnostics[bin_index]
        diagnostic["min_contact_propensity"] = (
            p_contact if diagnostic["min_contact_propensity"] is None
            else min(diagnostic["min_contact_propensity"], p_contact))
        diagnostic["min_response_propensity"] = (
            p_response if diagnostic["min_response_propensity"] is None
            else min(diagnostic["min_response_propensity"], p_response))
        diagnostic["max_weight"] = (
            weight if diagnostic["max_weight"] is None
            else max(diagnostic["max_weight"], weight))
    for bin_index, diagnostic in diagnostics.items():
        entries = [entry for entry in weighted_rows if entry[0] == bin_index]
        if entries:
            weights = np.array([entry[2] for entry in entries])
            diagnostic["ess"] = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
    return weighted_rows, diagnostics, contact_model, response_model


def _weighted_bin_means(weighted_rows, n_bins):
    """根据两阶段权重计算分桶调整均值。"""
    means = {}
    by_bin = defaultdict(list)
    for bin_index, outcome, weight, _, _ in weighted_rows:
        by_bin[bin_index].append((outcome, weight))
    for bin_index, entries in by_bin.items():
        values = np.array([outcome for outcome, _ in entries])
        weights = np.array([weight for _, weight in entries])
        means[bin_index] = float(np.sum(weights * values) / np.sum(weights))
    return means


def _delta_means(group_rows, deltas):
    """MNAR 边界分析（覆盖窗内合格总体）。

    对每个时间桶，已应答者保留实测值，其余合格 person-bin（未安排联系与
    联系失败）按组内观察均值加 delta 平移后单次插补，再对全部合格者求均值。
    delta=0 给出 MAR 下的朴素插补参照，便于与 IPW 对照。
    """
    observed = [row["outcome"] for row in group_rows if row["outcome"] is not None]
    reference = float(np.mean(observed)) if observed else None
    results = {}
    for delta in deltas:
        by_bin = defaultdict(lambda: [[], 0])
        for row in group_rows:
            if row["outcome"] is not None:
                by_bin[row["bin"]][0].append(row["outcome"])
            else:
                by_bin[row["bin"]][1] += 1
        bin_means = {}
        for bin_index, (observed_values, n_missing) in by_bin.items():
            if not observed_values or reference is None:
                bin_means[bin_index] = None
                continue
            fill = reference + delta
            total = observed_values + [fill] * n_missing
            bin_means[bin_index] = float(np.mean(total))
        results[_round(delta)] = bin_means
    return results


class _GroupArrays:
    """某个课程组内全部合格 person-bin 的列式表示，供 bootstrap 快速重抽样。"""

    def __init__(self, rows, n_bins):
        people = sorted({row["person"] for row in rows})
        person_index = {person: i for i, person in enumerate(people)}
        self.people = people
        self.person_id = np.array([person_index[row["person"]] for row in rows])
        self.rows_by_person = [np.where(self.person_id == i)[0] for i in range(len(people))]
        self.bin_index = np.array([row["bin"] for row in rows])
        self.contacted = np.array([row["contact_status"] != "not_contacted" for row in rows], dtype=float)
        self.responded = np.array([row["outcome"] is not None for row in rows], dtype=float)
        self.outcome = np.array([row["outcome"] if row["outcome"] is not None else np.nan
                                 for row in rows], dtype=float)
        ages = np.array([row["age"] for row in rows], dtype=float)
        prior = np.array([row["prior_mean"] if row["prior_mean"] is not None else np.nan
                          for row in rows])
        self.prior_missing = np.isnan(prior)
        # 缺失既往测量的填充值在全组合格面板上确定一次，子集模型复用同一值。
        self.prior_fill_value = float(np.nanmean(prior)) if (~self.prior_missing).any() else 0.0
        self.prior_filled = np.where(self.prior_missing, self.prior_fill_value, prior)
        self.n_contacts = np.array([row["n_contacts"] for row in rows], dtype=float)
        self.n_scheduled = np.array([row["n_scheduled"] for row in rows], dtype=float)
        self.n_prior_contacts = np.array([row["n_prior_contacts"] for row in rows], dtype=float)
        self.ages = ages
        self.n_bins = n_bins
        # 标准化基准（中心化位置与尺度）只在全组合格面板上计算一次，
        # 两阶段与点估计/bootstrap 共用，避免子集各自中心化造成路径不一致。
        self.scales = tuple(
            (float(values.mean()), float(values.std())) for values in (
                ages, self.prior_filled, self.prior_missing.astype(float),
                self.n_contacts, self.n_scheduled, self.n_prior_contacts)
        )

    def design(self, mask):
        return _design_matrix(
            self.ages[mask], self.prior_filled[mask], self.prior_missing[mask],
            self.n_contacts[mask], self.n_scheduled[mask],
            self.n_prior_contacts[mask], self.bin_index[mask], self.n_bins, self.scales,
        )[0]


def _bootstrap_estimate(arrays, mask, penalty):
    """一次 bootstrap 重抽样本上的两阶段稳定化 IPW 分桶均值。"""
    contact_target = arrays.contacted[mask]
    if (len(mask) < MIN_MODEL_ROWS or contact_target.sum() < MIN_MODEL_EVENTS
            or (1 - contact_target).sum() < MIN_MODEL_EVENTS):
        p_contact = np.full(len(mask), contact_target.mean())
    else:
        p_contact = _irls_probabilities(arrays.design(mask), contact_target, penalty)
    contacted_mask_local = contact_target == 1
    global_positions = mask[contacted_mask_local]
    response_target = arrays.responded[global_positions]
    if (len(global_positions) < MIN_MODEL_ROWS or response_target.sum() < MIN_MODEL_EVENTS
            or (1 - response_target).sum() < MIN_MODEL_EVENTS):
        p_response = np.full(len(global_positions), response_target.mean())
    else:
        p_response = _irls_probabilities(arrays.design(global_positions),
                                         response_target, penalty)
    bin_index = arrays.bin_index[mask]
    means = {}
    for bin_id in np.unique(bin_index):
        bin_mask = bin_index == bin_id
        eligible_n = bin_mask.sum()
        # 以下布尔数组都定义在完整重抽样本（长度 = len(mask)）上。
        answered_full = contacted_mask_local & (arrays.responded[mask] == 1)
        contacted_in_bin = contacted_mask_local & bin_mask
        contacted_n = contacted_in_bin.sum()
        answered_in_bin = answered_full & bin_mask
        answered_n = answered_in_bin.sum()
        if eligible_n == 0 or contacted_n == 0 or answered_n == 0:
            continue
        # 应答者在“被联系者”序列中的位置，用于取第二阶段概率。
        answer_local = np.where(answered_in_bin[contacted_mask_local])[0]
        contact_p = p_contact[answered_in_bin]
        response_p = p_response[answer_local]
        stable = ((contacted_n / eligible_n) / contact_p) * (
            (answered_n / contacted_n) / response_p)
        values = arrays.outcome[mask][answered_in_bin]
        means[int(bin_id)] = float(np.sum(stable * values) / np.sum(stable))
    return means


def _bootstrap_intervals(panel, groups, n_bins, settings, rng):
    """按人分层重抽的 bootstrap，得到两阶段 IPW 均值的百分位置信区间。"""
    intervals = {}
    for group in groups:
        group_rows = [row for row in panel if row["group"] == group]
        arrays = _GroupArrays(group_rows, n_bins)
        n_people = len(arrays.people)
        penalty = None
        draws = defaultdict(list)
        for _ in range(settings["bootstrap_replicates"]):
            picks = rng.choice(n_people, size=n_people, replace=True)
            # 重抽人即重抽其全部 person-bin 行，保持同一人内部的聚类结构。
            mask = np.concatenate([arrays.rows_by_person[pick] for pick in picks])
            if penalty is None:
                penalty = np.array([0.0] + [1.0] * (arrays.design(mask).shape[1] - 1))
            means = _bootstrap_estimate(arrays, mask, penalty)
            for bin_index, mean in means.items():
                draws[bin_index].append(mean)
        intervals[group] = {}
        for bin_index in range(n_bins):
            replicate_means = np.array(draws.get(bin_index, []))
            if len(replicate_means) >= 50:
                intervals[group][bin_index] = [float(np.percentile(replicate_means, 2.5)),
                                                float(np.percentile(replicate_means, 97.5))]
            else:
                intervals[group][bin_index] = None
    return intervals


def _period_status(diagnostic, estimate_available):
    """根据正值性与有效样本量给出时间桶的证据状态。"""
    if not estimate_available or diagnostic["n_answered"] == 0:
        return "insufficient_evidence"
    propensities = [value for value in (diagnostic["min_contact_propensity"],
                                        diagnostic["min_response_propensity"])
                    if value is not None]
    positivity_ok = (bool(propensities) and min(propensities) >= MIN_PROPENSITY
                     and diagnostic["max_weight"] is not None
                     and diagnostic["max_weight"] <= MAX_WEIGHT_RATIO)
    ess_ratio = diagnostic["ess"] / diagnostic["n_answered"]
    if diagnostic["n_answered"] < MIN_ANSWERED or ess_ratio < MIN_ESS_RATIO or not positivity_ok:
        return "conditional_only"
    return "comparable"


def _round(value, digits=4):
    return float(np.round(float(value), digits))


def visit_adjusted_trend(tables, parameters):
    settings = parse_parameters(parameters)
    panel, n_bins = _build_panel(tables, settings)
    if not tables["visits"]:
        return _empty_result(tables, settings, panel, n_bins)

    groups = sorted({row["group"] for row in panel})
    result = {
        "analysis": ANALYSIS,
        "parameters": settings,
        "estimand": "每个时间桶内仍处于个人观察窗口的学员平均生活体验（描述性走势，非课程效应）",
        "assumptions": [
            "MAR（两阶段）：在既往测量、联系节奏、预定标记、年龄与时间桶条件下，被联系与取得应答都与未观测体验独立",
            "正值性：各协变量层内被联系概率与应答概率都显著大于 0",
            "窗口外日志仅作历史资料，不进入估计；未安排联系、联系失败、离开服务范围分开处理",
        ],
        "groups": {},
        "interpretation": {
            "causal_claim": "调整后的曲线只描述群体体验随时间的描述性走势，不能解释为课程造成的改善；enrolled 是登记组别，不是随机分配",
            "evidence_status_legend": {
                "comparable": "应答人数、有效样本量与正值性诊断均达标，时间桶之间可作描述性比较",
                "conditional_only": "样本或正值性支撑不足，结论仅在 MAR 等假设成立时有效，需结合 delta 边界阅读",
                "insufficient_evidence": "没有可用本人测量，不给出估计值",
            },
            "sensitivity": "delta_adjusted_mean 假设未取得测量的合格 person-bin（未安排联系与联系失败）其潜在体验相对组内观察均值平移 delta 个评分单位，用于量化对未观测状态依赖（MNAR）的敏感程度；delta=0 是 MAR 参照",
        },
        "diagnostics": {
            "min_answered_threshold": MIN_ANSWERED,
            "min_ess_ratio": MIN_ESS_RATIO,
            "min_propensity": MIN_PROPENSITY,
            "max_weight_ratio": MAX_WEIGHT_RATIO,
        },
    }
    intervals = _bootstrap_intervals(panel, groups, n_bins, settings,
                                     np.random.default_rng(settings["seed"]))
    windows = {row["person"]: row for row in tables["windows"]}
    group_by_person = {row["person"]: int(row["enrolled"]) for row in tables["people"]}
    out_window_contacts = defaultdict(int)
    out_window_visits = defaultdict(int)
    for row in tables["contacts"]:
        window = windows[row["person"]]
        if not (window["start"] <= row["day"] < window["end"]):
            out_window_contacts[group_by_person[row["person"]]] += 1
    for row in tables["visits"]:
        window = windows[row["person"]]
        if not (window["start"] <= row["day"] < window["end"]):
            out_window_visits[group_by_person[row["person"]]] += 1
    for group in groups:
        rows = [row for row in panel if row["group"] == group]
        group_arrays = _GroupArrays(rows, n_bins)
        weighted_rows, diagnostics, contact_model, response_model = _two_stage_weights(
            rows, n_bins, group_arrays)
        adjusted = _weighted_bin_means(weighted_rows, n_bins)
        delta_results = _delta_means(rows, settings["deltas"])
        raw_means = {}
        for bin_index in range(n_bins):
            observed = [row["outcome"] for row in rows
                        if row["bin"] == bin_index and row["outcome"] is not None]
            raw_means[bin_index] = float(np.mean(observed)) if observed else None

        periods = []
        counts = {"eligible_person_bins": 0, "not_contacted_person_bins": 0,
                  "failed_contact_person_bins": 0, "answered_person_bins": 0,
                  "out_of_window_contacts": 0, "out_of_window_visits": 0}
        for bin_index in range(n_bins):
            bin_rows = [row for row in rows if row["bin"] == bin_index]
            diagnostic = diagnostics.get(bin_index, {
                "n_eligible": len(bin_rows), "n_contacted": 0, "n_answered": 0,
                "ess": 0.0, "min_contact_propensity": None,
                "min_response_propensity": None, "max_weight": None})
            estimate = adjusted.get(bin_index)
            status = _period_status(diagnostic, estimate is not None)
            counts["eligible_person_bins"] += len(bin_rows)
            counts["not_contacted_person_bins"] += sum(1 for row in bin_rows
                                                       if row["contact_status"] == "not_contacted")
            counts["failed_contact_person_bins"] += sum(1 for row in bin_rows
                                                        if row["contact_status"] == "failed")
            counts["answered_person_bins"] += diagnostic["n_answered"]
            period = {
                "bin": bin_index,
                "day_start": bin_index * settings["bin_days"],
                "day_end": (bin_index + 1) * settings["bin_days"],
                "eligible_people": len(bin_rows),
                "not_contacted_people": sum(1 for row in bin_rows
                                            if row["contact_status"] == "not_contacted"),
                "failed_contact_people": sum(1 for row in bin_rows
                                             if row["contact_status"] == "failed"),
                "answered_people": diagnostic["n_answered"],
                "raw_respondent_mean": _round(raw_means[bin_index]) if raw_means[bin_index] is not None else None,
                "adjusted_mean": _round(estimate) if estimate is not None else None,
                "confidence_interval_95": ([_round(intervals[group][bin_index][0]),
                                            _round(intervals[group][bin_index][1])]
                                           if intervals[group].get(bin_index) else None),
                "ess": _round(diagnostic["ess"], 2),
                "min_contact_propensity": (_round(diagnostic["min_contact_propensity"])
                                           if diagnostic["min_contact_propensity"] is not None else None),
                "min_response_propensity": (_round(diagnostic["min_response_propensity"])
                                            if diagnostic["min_response_propensity"] is not None else None),
                "max_weight": _round(diagnostic["max_weight"]) if diagnostic["max_weight"] is not None else None,
                "evidence_status": status,
                "delta_adjusted_mean": {str(delta): (_round(means.get(bin_index))
                                                     if means.get(bin_index) is not None else None)
                                        for delta, means in sorted(delta_results.items(), key=lambda item: item[0])},
            }
            periods.append(period)
        counts["out_of_window_contacts"] = out_window_contacts[group]
        counts["out_of_window_visits"] = out_window_visits[group]
        result["groups"][_group_name(group)] = {
            "enrolled": group,
            "contact_propensity_model": contact_model,
            "response_propensity_model": response_model,
            "n_people": len({row["person"] for row in rows}),
            "coverage": counts,
            "comparable_bins": [period["bin"] for period in periods
                                if period["evidence_status"] == "comparable"],
            "conditional_bins": [period["bin"] for period in periods
                                 if period["evidence_status"] == "conditional_only"],
            "insufficient_bins": [period["bin"] for period in periods
                                  if period["evidence_status"] == "insufficient_evidence"],
            "periods": periods,
        }
    return result


def _group_name(group):
    return "course_participants" if group == 1 else "non_participants"


def _empty_result(tables, settings, panel, n_bins):
    """没有任何本人测量的版本（如 limited）：只保留联系过程，不产生估计。"""
    groups = sorted({row["group"] for row in panel})
    windows = {row["person"]: row for row in tables["windows"]}
    group_by_person = {row["person"]: int(row["enrolled"]) for row in tables["people"]}
    out_window_contacts = defaultdict(int)
    for row in tables["contacts"]:
        window = windows[row["person"]]
        if not (window["start"] <= row["day"] < window["end"]):
            out_window_contacts[group_by_person[row["person"]]] += 1
    result = {
        "analysis": ANALYSIS,
        "parameters": settings,
        "estimand": "每个时间桶内仍处于个人观察窗口的学员平均生活体验（描述性走势，非课程效应）",
        "assumptions": ["当前资料版本没有任何本人测量，无法估计体验走势"],
        "groups": {},
        "interpretation": {
            "causal_claim": "调整后的曲线只描述群体体验随时间的描述性走势，不能解释为课程造成的改善；enrolled 是登记组别，不是随机分配",
            "evidence_status_legend": {
                "comparable": "应答人数、有效样本量与正值性诊断均达标，时间桶之间可作描述性比较",
                "conditional_only": "样本或正值性支撑不足，结论仅在 MAR 等假设成立时有效，需结合 delta 边界阅读",
                "insufficient_evidence": "没有可用本人测量，不给出估计值",
            },
            "sensitivity": "delta_adjusted_mean 假设未取得测量的合格 person-bin（未安排联系与联系失败）其潜在体验相对组内观察均值平移 delta 个评分单位，用于量化对未观测状态依赖（MNAR）的敏感程度；delta=0 是 MAR 参照",
        },
        "diagnostics": {"reason": "no_visits_in_release",
                        "min_answered_threshold": MIN_ANSWERED,
                        "min_ess_ratio": MIN_ESS_RATIO,
                        "min_propensity": MIN_PROPENSITY,
                        "max_weight_ratio": MAX_WEIGHT_RATIO},
    }
    for group in groups:
        rows = [row for row in panel if row["group"] == group]
        periods = []
        counts = {"eligible_person_bins": 0, "not_contacted_person_bins": 0,
                  "failed_contact_person_bins": 0, "answered_person_bins": 0,
                  "out_of_window_contacts": 0, "out_of_window_visits": 0}
        for bin_index in range(n_bins):
            bin_rows = [row for row in rows if row["bin"] == bin_index]
            counts["eligible_person_bins"] += len(bin_rows)
            counts["not_contacted_person_bins"] += sum(1 for row in bin_rows
                                                       if row["contact_status"] == "not_contacted")
            counts["failed_contact_person_bins"] += sum(1 for row in bin_rows
                                                        if row["contact_status"] == "failed")
            periods.append({
                "bin": bin_index,
                "day_start": bin_index * settings["bin_days"],
                "day_end": (bin_index + 1) * settings["bin_days"],
                "eligible_people": len(bin_rows),
                "not_contacted_people": sum(1 for row in bin_rows
                                            if row["contact_status"] == "not_contacted"),
                "failed_contact_people": sum(1 for row in bin_rows
                                             if row["contact_status"] == "failed"),
                "answered_people": 0,
                "raw_respondent_mean": None,
                "adjusted_mean": None,
                "confidence_interval_95": None,
                "ess": 0.0,
                "min_contact_propensity": None,
                "min_response_propensity": None,
                "max_weight": None,
                "evidence_status": "insufficient_evidence",
                "delta_adjusted_mean": {str(_round(delta)): None for delta in settings["deltas"]},
            })
        counts["out_of_window_contacts"] = out_window_contacts[group]
        counts["out_of_window_visits"] = 0
        result["groups"][_group_name(group)] = {
            "enrolled": group,
            "contact_propensity_model": "not_estimated",
            "response_propensity_model": "not_estimated",
            "n_people": len({row["person"] for row in rows}),
            "coverage": counts,
            "comparable_bins": [],
            "conditional_bins": [],
            "insufficient_bins": list(range(n_bins)),
            "periods": periods,
        }
    return result
