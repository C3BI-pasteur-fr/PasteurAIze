from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from .arxiv_search import ArxivSearch
from .google_scholar import GoogleScholar
import logging

# Enable logging at debug level
logging.basicConfig(level=logging.DEBUG)

# Enable HTTP client logging
import http.client as http_client


server = Server("mcp-scholarly")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        # types.Tool(
        #     name="search-arxiv",
        #     description="Search arxiv for articles related to the given keyword.",
        #     inputSchema={
        #         "type": "object",
        #         "properties": {
        #             "keyword": {"type": "string"},
        #         },
        #         "required": ["keyword"],
        #     },
        # ),
        types.Tool(
            name="retrieve-articles-link-google-scholar",
            description="Retrieve the link the articles on Google Scholar.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name != "retrieve-articles-link-google-scholar":
        raise ValueError(f"Unknown tool: {name}")

    if not arguments:
        raise ValueError("Missing arguments")

    keyword = arguments.get("keyword")
    logging.debug(f"Tool called: {name} with arguments: {arguments}")
    if not keyword:
        raise ValueError("Missing keyword")

    # Notify clients that resources have changed
    await server.request_context.session.send_resource_list_changed()
    formatted_results = []
    # if name == "search-arxiv":
    #     arxiv_search = ArxivSearch()
    #     formatted_results = arxiv_search.search(keyword)
    if name == "retrieve-articles-link-google-scholar":
        google_scholar = GoogleScholar()
        formatted_results = google_scholar.search_pubs(keyword=keyword)

    return [
        types.TextContent(
            type="text",
            text=f"Search articles for {keyword}:\n" + "\n\n\n".join(formatted_results),
        ),
    ]


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):

        http_client.HTTPConnection.debuglevel = 1
        logging.basicConfig(level=logging.DEBUG)
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-scholarly",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
