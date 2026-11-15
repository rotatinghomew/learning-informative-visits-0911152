# 学员回访分析资料约定

资料全部是本地合成业务记录，不含真实个人信息。`term-a` 为完整版本，`term-b` 保持业务值不变而调整表行次序，用于检验分析是否不恰当地依赖到达顺序；`limited` 保留登记资料并移除一类观测，表示证据不足的版本。版本内容不可就地修改。

- `people`：业务唯一键为 `person`。
- `contacts`：业务唯一键为 `person, day`。
- `visits`：业务唯一键为 `person, day`。
- `windows`：业务唯一键为 `person`。

day 是相对入组日的天数；windows 是服务可观察范围，范围外日志只作历史资料。contacts.contacted 表示已尝试联系，不代表本人回答；visits 存在才表示得到本人的测量，scheduled 区分预定回访。wellbeing 越大体验越好，mobility 越大出行越方便，均不是医学诊断。people.enrolled 为登记课程参与组，不是随机分配。

字段之外不能从示例的编号顺序或随机生成规律推导业务关系。涉及权重、相关范围、模型支撑的判断必须使用明确登记的字段和业务假设。

## visit_adjusted_trend 对字段的使用

`visit_adjusted_trend` 在 person × 时间桶（默认 30 天）面板上工作，分析单元是“某学员在某时间桶内是否被联系、是否取得本人测量”，而不是单条回访记录，因此同一个人的密集回访不会被当成多名独立学员。

- 窗口：只有 `windows.start <= day < windows.end` 的 person-bin 才进入估计；窗口外的 contacts/visits 只计入 `coverage.out_of_window_*` 作历史资料。`reason=移居` 等提前离开服务范围的人在离开后的桶里不再是合格总体。
- 未安排联系：合格 person-bin 在 contacts 中没有任何记录，记为 `not_contacted`，由第一阶段“被联系倾向”模型处理。
- 联系失败：contacts 有记录但 visits 没有对应测量，记为 `failed`，由第二阶段“取得应答倾向”模型处理。contacts.contacted 为真只表示尝试过。
- 应答：visits 存在才是本人测量；同桶多次测量取桶内均值，仍只占一个 person-bin。
- 分组：people.enrolled 区分课程参与者与未参加者，两组分别给出描述性走势；它不是随机分配，调整结果不能解释为课程效应。
- 协变量：倾向模型使用年龄、既往测量（含“无既往测量”指示）、桶内/既往联系次数、预定联系次数和时间桶。contacts.scheduled 只用于刻画联系安排，不代表应答。
- 敏感性：回访时机对未观测状态的依赖无法用现有字段检验，结果用 delta 边界（MNAR 情景）量化，并按应答人数、有效样本量（ESS）和倾向正值性把每个时间桶标为 comparable / conditional_only / insufficient_evidence。
