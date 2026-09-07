# 知识库测试语料 synthetic_v1

本数据集为人工设计与模板扩展的虚构测试资料，不是对项目真实功能、现实法规或实际业务数据的描述。

## 内容规模
新增58份UTF-8文档：24个项目各含需求说明和验收记录，共48份；制度与版本资料8份；长文档2份。
保留知识库原有 research_policy.md 和 project_notes.txt。

- projects：跨文档比较、相似主题干扰、表格数字、目标与实测区分。
- policies：历史版本、现行版本、未批准草案、例外条款和时间单位。
- long_docs：120个研究演练场景和240条事件记录，用于长文档分块、章节查询和范围读取。
- 中文为主，包含英文指标、字段名、事件编号及英文查询。

项目验收文件明确记录通过或未通过；不能从“能够运行”推断通过。
退款版本故意存在不同数值：必须结合批准状态、有效日期和适用对象判断。
长文档中的参数是场景局部设定，不可作为真实应用配置。
这是一套功能回归语料，模板化样本不能替代真实业务质量评测。

## 测试问题
queries.jsonl包含55个问题及预期答案，其中48个项目问题、7个版本/例外/长文档/无答案/跨语言问题。
每行字段：
- id：稳定题号。
- type：测试类别。
- question：只把此字段作为Agent输入。
- expected_answer：供人工或评测器核对，不作为Agent上下文。
- source_paths：相对于data/knowledge_base的预期来源。
- evidence_terms：用于检查语料完整性的关键文本，并非完整的自动评分规则。

评测资料放在data/evaluation/synthetic_v1，位于知识库之外。不要把evaluation目录加入入库根目录。

## 执行
在mini_research_agent项目根目录执行：

```powershell
uv run python -m mini_research_agent.scripts.ingest
uv run python -m mini_research_agent.cli
```

入库需要可用的Embedding模型，首次使用可能下载模型。单纯添加文件不会更新已有Chroma索引。
本次语料添加只验证Loader和Splitter；未自动运行真实Embedding入库，也未证明语义召回质量。

## 检查方法
1. 首次入库：检查报告failures为空且新增文档数量符合已有索引状态。
2. 原样再入库：应以unchanged_documents为主，chunks_written应为0。
3. 问答：从queries.jsonl选question，核对答案、适用范围和引用文件。
4. 长文档：使用DRILL-087与EVT-0195问题检查深层证据定位。
5. 无答案：确认没有编造上市日期或个人手机号。
6. 更新和删除：复制数据到临时目录并使用独立向量库后进行，不删除这套基准资料。

RAG知识库目录与受控文件工具的允许目录是独立配置。默认MCP_FILES_ROOT=files时，文件读取工具无法直接读取data/knowledge_base内的文档。需要联合验证时检查两者的目录授权是否一致；检索命中文档并不自动授予文件读取权限。
