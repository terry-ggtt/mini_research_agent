"""Prompt for final report generation."""

final_report_generation_prompt = """Based on all the research conducted, create a comprehensive, well-structured answer to the overall research brief:
<Research Brief>
{research_brief}
</Research Brief>

CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your response in English. If the user's messages are in Chinese, then MAKE SURE you write your entire response in Chinese.
This is critical. The user will only understand the answer if it is written in the same language as their input message.

Today's date is {date}.

Here are the findings from the research that you conducted:
<Findings>
{findings}
</Findings>

<Research Review>
Termination reason: {termination_reason}
Unresolved questions and evidence limitations:
{research_limitations}
</Research Review>
Distinguish evidence-supported conclusions, inferences requiring verification,
conflicting sources, and unresolved questions. Explicitly preserve the supplied
limitations. If the budget was exhausted or research could not continue, identify
which questions remain unanswered; do not merely claim research is complete.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. Cites web evidence with numeric citations, local knowledge-base evidence with its exact KB Citation ID, and MCP-read files with their exact FILE citation
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with separate web, knowledge-base, and MCP file subsections for the source types that are present

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Cite claims only with sources that appear in the provided findings.
- Assign each unique web URL a single numeric citation, such as [1].
- Cite knowledge-base evidence using the exact Citation ID supplied in the findings, such as [KB:chunk-id].
- Cite bounded file evidence using the exact FILE identifier supplied in the findings, such as [FILE:path/to/file#L10-L40].
- Preserve knowledge-base titles, source paths, sections, and page numbers whenever available.
- Preserve the exact source path for MCP-read files.
- Never invent a URL, local source path, Citation ID, section, page number, or source record.
- Never convert a local file path into a fabricated web link.
- Do not cite directory listings, allowed-directory listings, or filename-only search results as factual evidence.
- If web, knowledge-base, and MCP file evidence conflict, state the conflict and cite the conflicting sources.
- End with a ### Sources section.
- Use #### Web Sources, #### Knowledge Base Sources, and #### MCP File Sources subsections for the source types that are present.
- Number web sources sequentially without gaps.
- Web source format:
  [1] Source Title: URL
  [2] Source Title: URL
- Knowledge-base source format:
  [KB:chunk-id] Document Title
  Source: original/local/path
  Section: section path when available
  Page: page number when available
- MCP file source format:
  [FILE:path/to/file#L10-L40] Document Title when available
  Source: exact/path/passed/to/the/file/tool
  Lines: 10-40
- Omit any empty source subsection.
</Citation Rules>
"""
