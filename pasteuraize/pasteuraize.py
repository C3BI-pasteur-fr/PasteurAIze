from __future__ import annotations
import asyncio
from pathlib import Path

from PIL import Image

from transformers import AutoTokenizer

import pandas as pd
import streamlit as st
import logfire
from main_agent import main_agent, MessageHistory, resume_agent
from manage_schema import show_all_schemas, use_schema, show_current_schema

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolReturnPart,
)
import ast
import json

# Load environment variables if needed
from dotenv import load_dotenv

load_dotenv()
Image.MAX_IMAGE_PIXELS = 9999999999

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire="never")

# Define path to image assets
ASSETS_DIR = Path(__file__).parent / "assets"
USER_AVATAR_PATH = str(ASSETS_DIR / "user.png")
LOUIS_AVATAR_PATH = str(ASSETS_DIR / "louis.png")


def select_project():
    """
    Select the project to use.
    """
    projects = show_all_schemas()
    selected_project = st.sidebar.selectbox("Select a project", projects)
    use_schema(selected_project)


def create_chat():
    if st.session_state.key in st.session_state.messages:
        max_key = max(st.session_state.messages.keys())
        st.session_state.key = max_key + 1


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == "user-prompt":
        with st.chat_message("user", avatar=USER_AVATAR_PATH):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant", avatar=LOUIS_AVATAR_PATH):
            st.markdown(part.content)


def change_key(key):
    """
    set the session state key to the selected chat key,
    which will be used to display the messages in the main chat area.
    """
    st.session_state.key = key


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Run the agent in a stream
    try:
        history = MessageHistory(
            messages=st.session_state.messages[st.session_state.key]
        )
        async with main_agent.run_stream(
            user_input,
            message_history=st.session_state.messages[st.session_state.key][
                :-1
            ],  # pass entire conversation so far
            deps=history,  # pass the message history as a dependency
        ) as result:
            # We'll gather partial text to show incrementally
            partial_text = ""
            message_placeholder = st.empty()

            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                think = 0
                if "<think>" in partial_text:
                    # If we encounter a <think> tag, we split the text
                    think += 1
                if "</think>" in partial_text:
                    # If we encounter a </think> tag, we split the text
                    think -= 1
                    partial_text = partial_text.split("</think>")[1].strip()
                if think == 0:
                    message_placeholder.markdown(partial_text)
            # Add new messages from this run, excluding user-prompt messages
            filtered_messages = [
                msg
                for msg in result.new_messages()
                if not (
                    hasattr(msg, "parts")
                    and any(part.part_kind == "user-prompt" for part in msg.parts)
                )
            ]
            # Create a new list with modified messages
            new_filtered_messages = []
            all_new_messages = ""
            # print("Filtered messages:", filtered_messages, "\n\n --")
            for msg in filtered_messages:
                if isinstance(msg.parts[0], ToolReturnPart):
                    print("Found ToolReturnPart in message parts")
                    # print(st.session_state.data_run_history)
                    print("------------------------")
                    msg.parts[0].content = str(st.session_state.data_run_history)
                    new_filtered_messages.append(msg)
                    all_new_messages += str(msg)
                    st.session_state.data_run_history = {}
                elif (
                    hasattr(msg.parts[0], "content")
                    and "</think>" in msg.parts[0].content
                ):
                    print("Found </think> in message content")
                    # Extract content after </think>
                    after_think = msg.parts[0].content.split("</think>")[1]
                    if after_think.strip():  # Only keep non-empty content
                        msg.parts[0].content = after_think
                        new_filtered_messages.append(msg)
                        all_new_messages += str(msg)

                else:
                    new_filtered_messages.append(msg)
                    all_new_messages += str(msg)
                # print("New filtered messages:", new_filtered_messages, "\n\n --")
            filtered_messages = new_filtered_messages
            all_msg = []
            all_text = ""
            for msg in st.session_state.messages[st.session_state.key]:
                all_text += str(msg)
                all_msg.append(msg)
            token_size = len(tokenizer(all_text)["input_ids"])
            new_message_size = len(tokenizer(all_new_messages)["input_ids"])
            print("Token size of the entire conversation:", token_size)
            print("Token size of the new messages:", new_message_size)
            if token_size + new_message_size > 128000:
                print("Token size exceeds limit, filtering messages")
                resume = await resume_agent.run(
                    user_prompt=str(all_msg),
                )
                print("Resume agent response:", resume.output)
                st.session_state.messages[st.session_state.key] = []
                st.session_state.messages[st.session_state.key].append(
                    ModelResponse(
                        parts=[
                            TextPart(
                                content=json.dumps(
                                    {
                                        "resume of the previous exchanges": resume.output,
                                    }
                                )
                            )
                        ]
                    )
                )

                # Filter messages to keep only the last 10

            st.session_state.messages[st.session_state.key].extend(filtered_messages)

            # st.session_state.messages.extend(filtered_messages)

    finally:
        pass


def main():
    st.set_page_config(
        page_title="PasteurAIze", page_icon=":robot_face:", layout="centered"
    )
    st.title("PasteurAIze")
    st.subheader("Ask your questions to Louis")
    select_project()
    st.sidebar.title("History")
    st.sidebar.button(
        "New Chat", on_click=lambda: create_chat(), use_container_width=True
    )
    # Initialize chat history in session state if not present
    if "key" not in st.session_state:
        st.session_state.key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if st.session_state.key not in st.session_state.messages:
        st.session_state.messages[st.session_state.key] = []
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "data_run_history" not in st.session_state:
        # structure : {question: [(type, content),...],...} type can be : dataframe, vega_lite. content is the JSON string of the data.
        st.session_state.data_run_history = {}
        # Display all messages from the conversation so far
        # Each message is either a ModelRequest or ModelResponse.
        # We iterate over their parts to decide how to display them.

    for msg in st.session_state.messages[st.session_state.key]:
        # print("Displaying message:", msg)
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

        if isinstance(msg.parts[0], ToolReturnPart):
            print("Found ToolReturnPart History")
            # print(msg.parts[0])
            # print((msg.parts[0].content))
            all_data = ast.literal_eval(msg.parts[0].content)
            # print("All data from ToolReturnPart:", all_data)
            for data_list in all_data.values():
                # print("Processing data_list:", data_list)
                for data in data_list:
                    # print("Processing data:", data)
                    if data[0] == "dataframe":
                        print("TOP")
                        df = pd.DataFrame(json.loads(data[1]))
                        if len(df) == 1:
                            with st.chat_message(
                                "assistant",
                                avatar=LOUIS_AVATAR_PATH,
                            ):
                                st.write("Here is the answer to your question:")
                                st.write(df.iloc[0, 0])
                        if len(df) > 1:
                            print("history DataFrame found")
                            # st.data_editor(
                            #     df,
                            #     use_container_width=True,
                            #     key=str(uuid.uuid4()),
                            # )
                            st.dataframe(df, use_container_width=True)
                            print("DataFrame displayed in the UI")

                    elif data[0] == "vega_lite":
                        print("Found Vega-Lite spec")
                        vega_spec = json.loads(data[1])
                        print("type of vega_spec:", type(vega_spec))
                        print("Vega-Lite spec2 :", vega_spec, "\n\n")
                        with st.chat_message(
                            "assistant",
                            avatar=LOUIS_AVATAR_PATH,
                        ):
                            st.vega_lite_chart(
                                data=vega_spec,
                            )

    # Chat input for the user
    user_input = st.chat_input("What would you like to know")

    if user_input:
        print(st.session_state.key)
        # We append a new request to the conversation explicitly
        st.session_state.messages[st.session_state.key].append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        # Display user prompt in the UI
        with st.chat_message(
            "user",
            avatar=USER_AVATAR_PATH,
        ):
            st.markdown(user_input)
        # st.write(st.session_state.messages)
        # Display the assistant's partial response while streaming
        with st.chat_message(
            "assistant",
            avatar=LOUIS_AVATAR_PATH,
        ):
            # Actually run the agent now, streaming the text
            asyncio.run(run_agent_with_streaming(user_input))
            # st.write("Agent has finished running.")
            # st.write(st.session_state.messages[st.session_state.key])
            # After the agent has run, we can display the chart or DataFrame
        for msg in st.session_state.messages[st.session_state.key][
            len(st.session_state.messages[st.session_state.key]) - 2 :
        ]:
            # print("Processing message:", msg.parts[0])

            if isinstance(msg.parts[0], ToolReturnPart):
                print("Found ToolReturnPart for print")
                # print(msg.parts[0])
                # print((msg.parts[0].content))
                all_data = ast.literal_eval(msg.parts[0].content)
                for data_list in all_data.values():
                    for data in data_list:
                        if data[0] == "dataframe":
                            print("TOP")
                            df = pd.DataFrame(json.loads(data[1]))
                            if len(df) == 1:
                                with st.chat_message(
                                    "assistant",
                                    avatar=LOUIS_AVATAR_PATH,
                                ):
                                    st.write("Here is the answer to your question:")
                                    st.write(df.iloc[0, 0])
                            if len(df) > 1:
                                print("history DataFrame found")
                                # st.data_editor(
                                #     df,
                                #     use_container_width=True,
                                #     key=str(uuid.uuid4()),
                                # )
                                st.dataframe(df, use_container_width=True)
                                print("DataFrame displayed in the UI")

                        if data[0] == "vega_lite":
                            print("Found Vega-Lite spec")
                            vega_spec = json.loads(data[1])
                            print("Vega-Lite spec:", vega_spec)
                            st.vega_lite_chart(
                                data=vega_spec,
                            )

                # elif isinstance(msg, dict):
                #     # Assuming this is a Vega-Lite spec
                #     st.vega_lite_chart(
                #         spec=msg,
                #     )
                #     st.session_state.vega_lite_spec = msg
        # st.write(st.session_state.messages)
        # print(st.session_state.messages)
        # st.write(st.session_state.messages[-1].parts[0].content)
    # print("CHAT", st.session_state.messages)
    print("LENGTH", len(st.session_state.messages))
    for key, value in st.session_state.messages.items():
        print("key", key)
        print("len", len(value))
        if len(value) > 0:
            st.sidebar.button(
                value[0].parts[0].content[:30] + "...",
                use_container_width=True,
                on_click=lambda k=key: change_key(k),
                key=key,
                type="tertiary",
            )


if __name__ == "__main__":
    # asyncio.run(main())
    main()
