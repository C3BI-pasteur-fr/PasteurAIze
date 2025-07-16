import dotenv
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
import os
import streamlit as st
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import BM25ContentFilter


from pydantic_ai import ModelRetry

dotenv.load_dotenv()
SCHOLARLY_PATH = str(Path(__file__).resolve().parent)


scholarly = MCPServerStdio(
    "python",
    args=[
        "-c",
        f"import sys; sys.path.insert(0, '{SCHOLARLY_PATH}'); from mcp_scholarly import main; main()",
    ],
)
# f'import sys; sys.path.insert(0,{Path("__file__").parent} ); from mcp_scholarly import main; main()'
model = OpenAIModel(
    model_name=os.getenv("SEARCH_MODEL"),
    # model_name="mistral-small-2503-local",
    provider=OpenAIProvider(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    ),
)

search_agent = Agent(
    model=model,
    instrument=True,
    mcp_servers=[scholarly],
    system_prompt="""

You are a comprehensive Scientific Research Assistant specialized in biomedical and infectious disease research for Institut Pasteur. You operate in two clearly defined phases using tools.

## PHASE 1 - ARTICLE RESEARCH:
Use the `retrieve-articles-link-google-scholar` tool with the exact user query.
Once you have the results, format them as raw JSON with the following fields:

- **"link"**: The entire direct LINK to the article.
- **"title"**: The exact title of the article.
- **"user_query"**: The original user query. 

**Constraints**: Maximum 5 relevant results, strict JSON format.

## TRANSITION:
After completing Phase 1, you MUST EXPLICITLY proceed to Phase 2 without waiting for additional user input. Use the `get_paper_content` tool on the identified articles from Phase 1.

## PHASE 2 - ANALYSIS AND SYNTHESIS:
For articles from Phase 1, use the `get_paper_content` tool to retrieve their content.
Present your analysis in this structured format:

1. **Question Reformulation**: Begin by reformulating the question to confirm your understanding.
2. **Article Analysis**: Identify and analyze key findings, methodologies, and relevant conclusions.
3. **Information Synthesis**: Integrate information from different articles to formulate a comprehensive response.
4. **Clear Structure**: Organize your response logically with headings or bullet points if necessary.
5. **Conclusion**: Summarize the main points and offer additional perspectives.
6. **Citations**: Include the title, source, and URL of each article used.

## EXAMPLE WORKFLOW:
User query: "Latest findings on SARS-CoV-2 variants"
1. Search Google Scholar with this exact query
2. Format results as JSON
3. AUTOMATICALLY proceed to use `get_paper_content` on each article
4. Analyze and synthesize findings following the structured format

IMPORTANT: You MUST ALWAYS complete both phases for every request. The transition between phases should be automatic without requiring additional user prompting.
    """,
)


@search_agent.tool_plain(retries=3)
async def read_paper_content(papers, query):
    """Retrieve the content of academic papers found earlier with the google scholar search based on their direct links.
    Args:
        papers (list): A list of dictionaries containing paper metadata, including links and titles.
        query (str): The original search query used to find the papers.
    Returns:
        dict: A dictionary mapping links to their corresponding markdown content.
    """
    st.write(f"_I have found {len(papers)} relevant articles._")
    with st.spinner("_Retrieving content from the articles..._"):
        print("papers", papers)
        print("papers type", type(papers))
        if len(papers) == 0:
            raise ModelRetry("No papers found. Please search again ")
        else:
            for elt in papers:
                if "link" not in elt:
                    raise ModelRetry(
                        "No valid JSON found in the response. Search papers again."
                    )

                else:
                    done = True
        paper_md = {}
        browser_config = BrowserConfig(
            # use_managed_browser=True,
            # use_persistent_context=True,
            # user_data_dir="/Users/tperdere/Python_Project/SHAMA-MCP/FrenchVanna/crawler_session",
            headless=False,  # Set to False for debugging purposes
        )
        bm25_filter = BM25ContentFilter(
            user_query=query,
        )
        run_config = CrawlerRunConfig(
            # cache_mode=CacheMode.DISABLED,
            wait_for_images=True,
            exclude_external_links=True,
            remove_overlay_elements=True,
            process_iframes=True,
            markdown_generator=DefaultMarkdownGenerator(content_filter=bm25_filter),
            delay_before_return_html=0.5,
            scan_full_page=True,
            scroll_delay=0.5,
            magic=True,
        )  # Default crawl run configuration

        async with AsyncWebCrawler(config=browser_config) as crawler:

            for paper in papers:
                retry = 3
                done = False
                while retry > 0 and not done:
                    try:
                        link = paper["link"]
                        result = await crawler.arun(url=link, config=run_config)
                        md = result.markdown
                        if len(md.fit_markdown) > 10000:
                            md.fit_markdown = md.fit_markdown[:9000] + "\n\n[...]\n\n"
                        paper_md[link] = {paper["title"]: md.fit_markdown}
                        done = True
                    except KeyError:
                        retry -= 1
                        if retry == 1:
                            st.write(
                                f"Failed to retrieve content for link: {link}. Please check the link or try again later."
                            )
                            break

            # print(result.markdown)  # Print clean markdown content
            # await crawler.aclear_cache()
        # print(paper_md)

    return paper_md


async def run_search_agent(query: str) -> str:
    """Run the search agent with the provided query."""
    # Run the search agent with the given query
    with st.spinner("**Searching for relevant articles...**"):
        async with search_agent.run_mcp_servers():
            print("Running search agent with query:", query)
            response = await search_agent.run(
                user_prompt=query,
                message_history=st.session_state.messages[st.session_state.key][
                    len(st.session_state.messages[st.session_state.key]) - 5 : -1
                ],
                retries=3,
            )
            return response.output
