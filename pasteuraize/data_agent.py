import os

import logfire

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent
from pydantic_ai import ModelRetry

from dataclasses import dataclass

from openai import OpenAI

from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore
from vanna.openai.openai_chat import OpenAI_Chat

from manage_schema import show_current_schema

from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext

import streamlit as st


import pandas as pd

import json

import re

from transformers import AutoTokenizer

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model_name = os.getenv("DATA_MODEL")
vega_model = os.getenv("VEGA_MODEL")
# model_name = os.getenv("MODEL_NAME")
vanna_model = os.getenv("VANNA_MODEL")
db = os.getenv("DB_NAME")

client = OpenAI(api_key=api_key, base_url=base_url)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)


def setup_vanna():
    project_path = "Project/" + show_current_schema()
    vn = MyVanna(config={"model": vanna_model, "path": project_path})
    vn.connect_to_postgres(
        host="localhost",
        port="5432",
        dbname=db,
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    return vn


@dataclass
class Deps:
    vanna: MyVanna
    history: list = None
    # user: User -> todo # add user system later


model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(
        api_key=api_key,
        base_url=base_url,
    ),
)


data_agent = Agent(
    model=model,
    name="Data Agent",
    system_prompt="""
You are a biological data tool orchestrator at Institut Pasteur, responsible for executing appropriate data retrieval or visualization tools in response to research queries. You do NOT generate data, visualizations, code or new queries - you ONLY execute the tool(s) to answer the query provided.

CONTEXT: You operate within Institut Pasteur's computational infrastructure as a simple pass-through interface. You handle genomic sequences, protein structures, epidemiological datasets, microscopy data, and various omics data (transcriptomics, proteomics, etc.) common in Institut Pasteur's research.

OBJECTIVE: Your sole purpose is to:
- Call only the appropriate tool(s) based on the research query
- Return ONLY the tool's output without modification
- Confirm successful tool execution with a simple standardized message
- If you have to make several calls to tools you return the output of the last call only

PROCESS:
1. Determine which tool(s) is/are required based on the query provided.
2. Execute the tool(s) with the appropriate parameters:
    - If the query requires data retrieval, use the `ask_database` tool 
    - If the query requires a plot, a chart or any kind of visualization, use the `generate_visualization` tool. 
    - If the query is not clear or cannot be processed, return a standardized error message
3. After the final tool execution, provide ONLY a simple confirmation message

FORMAT FOR CONFIRMATION MESSAGES:
- For data retrieval: "Data retrieval tool executed successfully. Data has been retrieved and displayed to the user."
- For visualization: "Visualization tool executed successfully. Visualization has been displayed to the user."
- For errors: "Unable to process the query as provided. Please try again with a different query or provide more information."


IMPORTANT: If the user does not specifically request a visualization, do not provide one. Your role is to execute tool(s) as instructed, then stop. The actual data and visualizations are handled and displayed by the tools themselves to the user through an interface.Do NOT attempt to generate or modify any data, visualizations, or code on your own. You are a simple pass-through interface for executing tools.""",
    deps_type=Deps,
    retries=3,
)


vega = OpenAIModel(
    model_name=vega_model, provider=OpenAIProvider(api_key=api_key, base_url=base_url)
)
vega_agent = Agent(
    model=vega,
    name="Vega Agent",
    system_prompt="""## Role

You are an expert in scientific data visualization specializing in generating **Vega-Lite specifications** at the **Institut Pasteur**.

---

## CONTEXT

At the Institut Pasteur, we work with DataFrames containing biological data (genomic sequencing, proteomics, epidemiological data) that require precise and informative visualizations.

---

## OBJECTIVE

Generate ONLY complete **Vega-Lite JSON specifications** that:

1. Include the `"data"` section configured to accept an external DataFrame  
2. Are optimized for effectively visualizing biological data  
3. Allow easy customization of visual parameters  

---

## PROCESS

1. Analyze the visualization request  
2. Create a complete Vega-Lite specification including the appropriate `"data"` structure  
3. Do **not** use `aggregate` or `fold`  

---

## FORMAT

- The specification must be in a well-formatted JSON code block  
- Use explicit variable names  

---

## STRUCTURE EXAMPLES

For a DataFrame, the `"data"` section might look like:
"data": {
  "values": [...] // This is where the DataFrame will be inserted
}
""",
)


@data_agent.tool(retries=3, strict=False)
async def ask_database(ctx: RunContext[Deps], question: str) -> str:
    """This tool retrieves data from the database using Vanna.
    It generates an SQL query based on the user's question, checks if the SQL is valid, and then executes it to retrieve the data.
    If the SQL query is not valid or execution fails, it raises a ModelRetry exception with an appropriate message.
        Args:
            question (str): The user's question to retrieve data from the database.
        Returns:
            str: A confirmation message indicating successful data retrieval with the SQL query used and if the data is not too large, the data.
    """
    with st.spinner("_Retrieving data from the database..._"):
        ctx.deps.vanna = setup_vanna()
        sql = ctx.deps.vanna.generate_sql(question=question, allow_llm_to_see_data=True)
        if not ctx.deps.vanna.is_sql_valid(sql=sql):
            raise ModelRetry(message="The generated SQL is not valid. Retry.")
        try:
            result = ctx.deps.vanna.run_sql(sql=sql)
        except Exception as e:
            raise ModelRetry(
                message=f"Unable to execute SQL query due to: {str(e)}. Retry by correcting inserting at the end of the argument 'question' :  'Please correct the SQL query that failed beceause of this error {str(e)}.'"
            )
        token_size = len(
            tokenizer(json.dumps(result.to_dict(orient="records")))["input_ids"]
        )
        print("Token size of the result:", token_size)
        if len(result) < 125000:  # let 3k tokens for the model to answer
            # print(f"Data retrieved successfully. The searched data is {result}")
            if not question in st.session_state.data_run_history:
                st.session_state.data_run_history[question] = [
                    ("dataframe", json.dumps(result.to_dict(orient="records")))
                ]
            else:
                st.session_state.data_run_history[question].append(
                    ("dataframe", json.dumps(result.to_dict(orient="records")))
                )
            # print(
            #     "st.session_state.data_run_history", st.session_state.data_run_history
            # )
            st.write("_Data retrieved successfully._")
            return f"Data retrieved successfully. The searched data is {json.dumps(result.to_dict(orient='records'))}."
        else:
            # print(
            #     "st.session_state.data_run_history", st.session_state.data_run_history
            # )
            st.error("_Max token lenght reached please search for something else_")
            return f"Data failed to be retrieved. The data is too large to be displayed. Please refine your query to retrieve a smaller dataset. The SQL query used was: {sql}. The data size is {len(result)} rows, which exceeds the limit of the model."


@data_agent.tool_plain(retries=3, strict=False)
async def generate_visualization(df, vizualisation: str) -> str:
    """This tool generates a visualization.
    It generates a visualization displayed to the user by producing a vega-lite JSON specification.
    If the visualization generation fails, it raises a ModelRetry exception with an appropriate message.
        Args:
            df : The dictionary representation of the dataframe to visualize in a list.
            vizualisation (str): The description of the vizualisation wanted, such as "bar chart of X vs Y" or "line chart of Z over time". As plain text.
        Returns:
            str: A confirmation message indicating successful visualization generation.
    """
    with st.spinner("_Generating visualization..._"):
        # df = st.session_state.dataframe
        print("df type", type(df))
        print(st.session_state.data_run_history)
        if st.session_state.data_run_history is None:
            df = pd.DataFrame(df)
        else:
            for key in st.session_state.data_run_history:
                if "dataframe" in st.session_state.data_run_history[key][0]:
                    df = json.loads(st.session_state.data_run_history[key][0][1])

                    break
        print("df content", df)
        if df is None or len(df) == 0:
            raise ModelRetry(
                message="No data available to generate a visualization. Please run a data retrieval query first."
            )
        df = pd.DataFrame(df)
        print("df type", type(df))
        head = df.head(n=3)

        head = head.to_json()
        # pattern = r"```json\s*({.*?})\s*```"
        pattern = r"({.*})"
        # Prompt
        prompt = f"""{head}, {vizualisation}"""
        # Run the agent to get the Vega-Lite specification
        response = await vega_agent.run(user_prompt=prompt)
        # st.write("Response from agent:", response.output)
        # Extract the JSON from the response
        question = f"here is a vega-lite specification for a visualization; correct it if needed {response.output}. Only provide the JSON and NOTHING ELSE, NO SMALL TALK."

        response = await vega_agent.run(user_prompt=question)
        print("Corrected:", response.output)
        match = re.search(pattern, response.output, re.DOTALL)

        if match:
            json_str = match.group(1)
            print("Extracted JSON:", json_str)
            print("type of json_str", type(json_str))
        else:
            raise ValueError(
                "No valid JSON found in the response. Answer :", response.output
            )
        try:
            update_vega_lite_spec = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ModelRetry(
                f"Error decoding JSON: {e}; please check the Vega-Lite specification."
            )

        update_vega_lite_spec["data"] = {"values": df.to_dict(orient="records")}
        print("Vega-Lite specification:", update_vega_lite_spec)
        if not vizualisation in st.session_state.data_run_history:
            st.session_state.data_run_history[vizualisation] = [
                (
                    "vega_lite",
                    json.dumps(update_vega_lite_spec),
                )
            ]
        else:
            st.session_state.data_run_history[vizualisation].append(
                ("vega_lite", json.dumps(update_vega_lite_spec))
            )
    st.write("_Vizualization generated successfully._")
    return f"Visualization generated successfully. It has already been displayed to the user."


async def run_data_agent(question: str, deps: Deps):
    deps.vanna = setup_vanna()
    result = await data_agent.run(
        user_prompt=question,
        deps=deps,
        message_history=st.session_state.messages[st.session_state.key][
            len(st.session_state.messages[st.session_state.key]) - 5 : -1
        ],
    )
    return result
