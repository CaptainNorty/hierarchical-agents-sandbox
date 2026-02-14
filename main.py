from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

load_dotenv()


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 72F and sunny in {city}"


agent = create_react_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[get_weather],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Salt Lake City?"}]}
)

for msg in result["messages"]:
    print(f"{msg.type}: {msg.content}")
