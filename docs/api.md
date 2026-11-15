# 分析服务接口

资料服务在 8765 端口提供 `/releases`、`/releases/term-a` 和版本内的表分页。表页携带 revision；客户端必须读取同一清单版本并验证摘要，不能合并不同版本的页。

应用服务在 8766 端口提供 `GET /catalog`、`POST /analyses`、`GET /analyses`、`GET /analyses/{id}`。当前目录登记了 overview 课程资料汇总与 visit_adjusted_trend 回访调整走势。分析实现登记在 `learning.domain.ANALYSES`，函数接收完整 tables 和 parameters，返回可序列化对象。

## visit_adjusted_trend

创建请求示例：

```json
{
  "request_key": "half-year-trend",
  "release": "term-a",
  "analysis": "visit_adjusted_trend",
  "parameters": {
    "bin_days": 30,
    "horizon_days": 180,
    "bootstrap_replicates": 1000,
    "seed": 20261115,
    "deltas": [-10.0, 0.0, 10.0]
  }
}
```

parameters 全部可选；省略时使用默认值。`bin_days`（1–180 的整数）与 `horizon_days`（bin_days 的整数倍）决定时间桶划分；`bootstrap_replicates`（100–20000）是按人重抽的分层 bootstrap 次数，`seed` 固定随机序列，`deltas` 是 1–9 个 MNAR 平移量，绝对值不超过 100。参数非法返回 422。

result 的结构与语义：

- `estimand`、`assumptions`：估计目标（窗内合格学员的群体平均体验，描述性、非课程效应）与两阶段 MAR/正值性假设。
- `groups.course_participants` 与 `groups.non_participants`：按 people.enrolled 分开的两组，各自带 `coverage`（合格/未安排联系/联系失败/应答 person-bin 数，以及窗口外日志数）、倾向模型名称、可比较与条件性时间桶清单。
- 每个 `periods[]` 元素是一个时间桶：`eligible_people`、`not_contacted_people`、`failed_contact_people`、`answered_people` 给出三类选择过程的人数；`raw_respondent_mean` 是仅应答者的原始均值（对照用），`adjusted_mean` 是两阶段稳定化 IPW 调整均值，`confidence_interval_95` 是按人 bootstrap 的百分位区间，`ess` 是有效样本量，`min_contact_propensity`/`min_response_propensity`/`max_weight` 是正值性诊断。
- `evidence_status`：`comparable` 表示样本量、ESS 与正值性均达标，可在桶间作描述性比较；`conditional_only` 表示结论只在 MAR 等假设成立时有效，必须结合 delta 边界阅读；`insufficient_evidence` 表示没有本人测量，不产出估计（如 limited 版本）。
- `delta_adjusted_mean`：每个 delta 下对全部未测量合格 person-bin 插补后的桶均值，量化对未观测状态依赖（MNAR）的敏感程度；delta=0 是 MAR 参照。
- `interpretation.causal_claim` 明确：调整曲线不能被表述为课程造成的改善。

创建请求字段为 request_key、release、analysis 和可选的 parameters 对象。request_key 是调用方的幂等编号：相同完整请求返回同一分析，相同编号不同内容返回 409。分析返回 id、request、revision 和 result。不同 request_key 是不同留存记录。缺少字段、未知分析类型或无效资料返回 422；查询不存在的记录返回 404。JSON 不允许 NaN 或 Infinity，无法估计的量使用 null 并在结果中说明原因。

保存后的分析内容不因资料更新而重算。新分析可在 result 内保存统计对象、参数、诊断及所需的版本化结构；不可覆盖其他分析或跳过资料完整性校验。数据库只保存本地分析，不是资料服务的业务真源。
