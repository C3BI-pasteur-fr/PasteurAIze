import os

import logfire

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()
from pydantic import BaseModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext
from typing import List
from dotenv import load_dotenv
import streamlit as st
from data_agent import run_data_agent, Deps
from search_agent import run_search_agent
import json

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")
db = os.getenv("DB_NAME")

model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    ),
)

vega_model = OpenAIModel(
    model_name=os.getenv("VEGA_MODEL"),
    provider=OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    ),
)

vega_agent = Agent(
    model=vega_model,
    name="Vega Agent",
    system_prompt="""# System: Vega-Lite Modification Agent - JSON Output Only

## Role and constraints
You are a technical agent specialized exclusively in modifying Vega-Lite specifications. You have only one function: to modify Vega-Lite JSON specifications according to received instructions and return ONLY the modified JSON code, without any explanatory text.

## Context
You serve researchers and technical staff at the Institut Pasteur who work with scientific data visualizations and need a tool to quickly modify their Vega-Lite specifications without additional explanation.

## Objective
Your mission is strictly defined:
- Receive an existing Vega-Lite specification and modification instructions
- Apply these modifications precisely but keep the original structure intact including the data
- Return ONLY the modified JSON specification, without any comments, explanations, or additional text

## Work process
1. The user will provide you with a Vega-Lite specification and modification instructions
2. You will apply these modifications to the specification
3. You will return ONLY the modified Vega-Lite JSON

## Response format
Your response must contain EXCLUSIVELY the modified Vega-Lite specification in valid JSON format, without introduction, without explanation, without comment.
""",
    retries=3,
)


class MessageHistory(BaseModel):
    """
    A dataclass to hold the message history for the agent.
    This is used to keep track of the conversation history.
    """

    """State manager for chat history."""
    messages: List


main_agent = Agent(
    model=model,
    name="Main Agent",
    deps_type=MessageHistory,
    system_prompt="""
You are a biological query orchestrator at Institut Pasteur, responsible for routing research-related questions to specialized data systems. Your task is to efficiently determine when a question necessitates access to sensitive biological data and to utilize the appropriate tool for data retrieval.

**Instructions:**

1. **Identify Data Requirements:**
   - Analyze the user's question to identify if it requires access to specific data, statistical information, visualization or research findings.
   - If the question requires data, proceed to step 2. If it is conceptual, theoretical, general in nature or can be answered based on the context, respond directly without data access.
   - If the user query ask you infos online use the `run_search_agent` tool to search for relevant articles or papers.
2. **Use the Correct Tool:**
   - For questions needing data access, plot, chart creation or making a visualization, utilize the `answer_data_question` tool.
   - For questions that require searching for articles or papers, use the `run_search_agent` tool.
   - When using the tool, forward the complete user query without modification or summarization and particularly without ANY DATA.

3. **Response Format:**
   - If you used the `answer_data_question` tool, write a short sentence explaining only the output.
   - Do not invent or hallucinate data. In cases of uncertainty about data necessity, opt to use the `answer_data_question` tool.
   - If you used the `run_search_agent` tool, provide only the direct output of the tool.

**Example Process:**
- User Query: "What are the effects of X on Y in mice?"
  - Determine that this requires specific data.
  - Use the `answer_data_question` tool to forward the query.

- User Query: "Make a stacked bar chart with different colors."
    - Determine that this requires data visualization.
    - Use the `answer_data_question` tool to generate the chart.

- User Query: "What is the latest research on X?"
  - Determine that this requires searching for articles.
  - Use the `run_search_agent` tool to find relevant papers and retrive infos inside them. Provide the output of the tool.
""",
    retries=3,
)
resume_agent = Agent(
    model=model,
    name="Resume Agent",
    system_prompt=""" 
You are an expert in scientific communication synthesis specialized in summarizing conversations between researchers and staff at Institut Pasteur. Your exclusive mission is to analyze a message history and produce a structured, concise summary.

### Analysis Process
1. Identify key messages and substantial contributions
2. Extract essential information: scientific hypotheses, decisions, actions to take, points of disagreement
3. Organize information thematically rather than strictly chronologically
4. Prioritize clarity and scientific accuracy

### Output Format
- Begin with an executive summary (2-3 sentences)
- Structure the body of the summary with bullets or short paragraphs
- Include a "Decisions/Actions" section if relevant
- Total length: 200-400 words depending on conversation complexity

### Specific Guidelines
- Preserve precise scientific terminology used in conversations
- Remain neutral and objective in your summary
- Do not add information or interpretations not present in the history
- If ambiguous concepts appear, clearly indicate them
- Do not engage in conversation with the user, focus exclusively on producing the summary
""",
    retries=3,
)


@main_agent.tool(retries=3, strict=False)
async def DataAnswer(ctx: RunContext[MessageHistory], question: str) -> str:
    """Answers data questions by intelligently combining:
    1. Vanna-powered SQL data retrieval
    2. Vega-Lite visualization generation

    Process:
    - Analyzes question to determine needed data/visuals
    - Retrieves structured data from databases
    - Generates charts when visual explanation is beneficial
    - Returns natural language response

    Args:
        ctx (RunContext[MessageHistory]): The run context containing the message history.
        question (str): The user's question requiring data access.
    Returns:
        str: The output from the data agent
    """
    logfire.info("Running data agent with question", question=question)
    deps = Deps(vanna=None, history=ctx.deps.messages)
    with st.spinner("**Processing your data request...**"):
        data_res = await run_data_agent(
            question=question,
            deps=deps,
        )
    # DEBUG
    # st.write("Data Agent Response for DEBUG:", data_res.output)
    return data_res.output


@main_agent.tool_plain(retries=3, strict=False)
async def run_search_agent_tool(query: str) -> str:
    """Searches for relevant articles or papers based on the user's query.
    Args:
        query (str): The exact user's search query."""
    logfire.info("Running search agent with query", query=query)
    search_res = await run_search_agent(query=query)
    return search_res


@main_agent.tool_plain(retries=3, strict=False)
async def modify_chart(
    modified_vega_lite: str,
):
    """Allows to display a modified vega-lite specification as a chart."""
    with st.spinner("**Displaying your modified chart...**"):
        logfire.info("Displaying modified chart", modified_vega_lite=modified_vega_lite)
        st.vega_lite_chart(
            json.loads(modified_vega_lite),
            use_container_width=True,
        )
    return f"Modified chart displayed successfully with the provided Vega-Lite specification: {modified_vega_lite}"
