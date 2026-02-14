from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's 72F and sunny in {city}"

def get_forecast(city: str) -> str:
    """Get the forecast for the upcoming three days in a given city."""
    return f"Forecast for {city}: Sunday: 53F low, 74F high. Monday: 51F low, 72F high. Tuesday: 55F low, 75F high."

def get_alerts(city: str) -> str:
    """Get important alerts about the weather for a given city."""
    return f"No urgent alerts in {city}"

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Converts a temperature value from one unit to another. Supported units: fahrenheit, celsius, kelvin."""
    f = from_unit.lower()
    t = to_unit.lower()

    if f == t:
        result = value
    elif f == "fahrenheit" and t == "celsius":
        result = (value - 32) * 5 / 9
    elif f == "celsius" and t == "fahrenheit":
        result = value * 9 / 5 + 32
    elif f == "fahrenheit" and t == "kelvin":
        result = (value - 32) * 5 / 9 + 273.15
    elif f == "kelvin" and t == "fahrenheit":
        result = (value - 273.15) * 9 / 5 + 32
    elif f == "celsius" and t == "kelvin":
        result = value + 273.15
    elif f == "kelvin" and t == "celsius":
        result = value - 273.15
    else:
        return f"Unknown conversion: {from_unit} to {to_unit}"

    return f"{value} {from_unit} = {result:.1f} {to_unit}"



def main():
    weather_agent = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[get_weather, get_forecast, get_alerts],
        name="weather_agent",
    )

    conversion_agent = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[convert_units],
        name="conversion_agent",
    )

    def ask_weather_agent(request: str) -> str:
        """Ask the weather agent for current weather, forecasts, or alerts for a city."""
        response = weather_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return response["messages"][-1].content

    def ask_conversion_agent(request: str) -> str:
        """Ask the conversion agent to convert a temperature between units (fahrenheit, celsius, kelvin)."""
        response = conversion_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return response["messages"][-1].content

    supervisor = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[ask_weather_agent, ask_conversion_agent],
        name="supervisor",
        system_prompt=(
            "You are a helpful assistant that coordinates specialized agents. "
            "Use ask_weather_agent for weather info (current conditions, forecasts, alerts). "
            "Use ask_conversion_agent to convert temperatures between units. "
            "You can call multiple agents to fulfill a request."
        ),
    )

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Salt Lake City in Celsius?"}]}
    )

    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()
