# Mini Deep Research Agent

## 目标

将用户的自然语言研究需求转换成结构化研究简报，
使用网络搜索收集资料，最终生成压缩后的研究结果。

## 场景一：需求不清楚

输入：

研究最好的咖啡店。

预期：

- Agent 判断信息不足。
- Agent 返回一个澄清问题。
- 不执行网络搜索。
- 不生成研究简报。

## 场景二：需求清楚

输入：

研究 2025 至 2026 年旧金山最好的精品咖啡店，
以咖啡豆、烘焙、冲泡技术和顾客评价作为标准。

预期：

- Agent 不再追问。
- Agent 生成 research_brief。
- research_brief 保留地点、时间和评价标准。

## 场景三：执行研究

输入：

一个有效的 research_brief。

预期：

- Agent 生成搜索词。
- Agent 调用搜索工具。
- Agent 读取搜索结果。
- Agent 判断是否继续搜索。
- 最多执行五轮工具调用。
- 最后生成 compressed_research。

## 场景四：工具失败

输入：

搜索工具发生超时或 API 错误。

预期：

- Graph 不崩溃。
- 错误被转换成 ToolMessage。
- Agent 可以重新搜索或结束研究。

## 最终输出

- research_brief
- compressed_research
- raw_notes
- sources