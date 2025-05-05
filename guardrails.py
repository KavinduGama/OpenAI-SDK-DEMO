#two types of guardrails are there 
#1. Input guardrails 2. Output guardrails
import os
from getpass import getpass
from agents import Agent, Runner
from pydantic import BaseModel

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")


class GuardrailOutput(BaseModel):
    is_triggered: bool
    reasoning: str

politics_agent = Agent(
    name = "Politics Check",
    instructions="Check if the user is asking you about political opinions",
    model = "gpt-4o-mini",
    output_type=GuardrailOutput
)

query = "What do you think about the labour party in the UK?"

async def main():
    result = await Runner.run(
        starting_agent = politics_agent,
        input = query
    )
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())