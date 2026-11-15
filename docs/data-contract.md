# 学员回访分析资料约定

资料全部是本地合成业务记录，不含真实个人信息。`term-a` 为完整版本，`term-b` 保持业务值不变而调整表行次序，用于检验分析是否不恰当地依赖到达顺序；`limited` 保留登记资料并移除一类观测，表示证据不足的版本。版本内容不可就地修改。

- `people`：业务唯一键为 `person`。
- `contacts`：业务唯一键为 `person, day`。
- `visits`：业务唯一键为 `person, day`。
- `windows`：业务唯一键为 `person`。

day 是相对入组日的天数；windows 是服务可观察范围，范围外日志只作历史资料。contacts.contacted 表示已尝试联系，不代表本人回答；visits 存在才表示得到本人的测量，scheduled 区分预定回访。wellbeing 越大体验越好，mobility 越大出行越方便，均不是医学诊断。people.enrolled 为登记课程参与组，不是随机分配。

字段之外不能从示例的编号顺序或随机生成规律推导业务关系。涉及权重、相关范围、模型支撑的判断必须使用明确登记的字段和业务假设。
