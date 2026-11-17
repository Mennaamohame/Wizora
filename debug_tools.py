
from langchain_core.messages import HumanMessage, ToolMessage
from agent import get_agent


agent = get_agent("gemini-3.5-flash-lite")


def test_tool(question, thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    for update in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates"
    ):

        for node, data in update.items():

            print("\nNODE:", node)

            for message in data.get("messages", []):

                if getattr(message, "tool_calls", None):
                    for call in message.tool_calls:
                        print("TOOL REQUEST:", call["name"])
                        print("ARGUMENTS:", call["args"])

                if isinstance(message, ToolMessage):
                    print("TOOL EXECUTED:", message.name)
                    print("TOOL RESULT:", message.content)

                elif getattr(message, "content", None):
                    if message.content:
                        print("MESSAGE:", message.content)


if __name__ == "__main__":

    test_tool(
        "Use the calculator tool only. Calculate 1287 multiplied by 349.",
        "qa-calculator-001"
    )

    test_tool(
        "Use web_search tool only. Search official python.org for the latest stable Python release.",
        "qa-search-001"
    )
