import os 
import asyncio
from getpass import getpass
from agents import Agent, Runner, function_tool
from openai.types.responses import ResponseFunctionCallArgumentsDeltaEvent, ResponseTextDeltaEvent

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

#01.Create a simple agent
agent = Agent(
    name="Assistant",
    instructions="You're a helpful assistant",
    model = "gpt-4o-mini"
)

# #02.Create a agent with a tool
@function_tool
def multiply(x:float, y:float) -> float:
    """Multiplies 'x' and 'y' to provide a precise answer"""
    return x * y 

agent_tool = Agent(
    name="Assistant",
    instructions=(
        "You're a helpful assistant. remember to always"
        "use the provided tools whenever possible. Do not"
        "rely on your own knowledge too much and instead"
        "use your tools to help you answer queries"
    ),
    model = "gpt-4o-mini",
    tools=[multiply]
)

class AgentDemo:
    def __init__(self, agent, agent_tool):
        self.agent = agent
        self.agent_tool = agent_tool

    async def run_simple_agent(self):
        print("***********Simple agent call - No streaming")
        result = await Runner.run(
            starting_agent=self.agent,
            input="what is the central park of the united states"
        )
        print(result.final_output)
        print("---------------------------------------------------------------------")
        print()

    async def run_simple_agent_streaming(self):
        print("***********Simple agent call - Streaming")
        response = Runner.run_streamed(
            starting_agent=self.agent,
            input="Tell me a story"
        )

        async for event in response.stream_events():
            if event.type == "raw_response_event" and \
                isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
        
        print("---------------------------------------------------------------------")
        print()

    async def run_tool_agent(self):
        print("***********Tool agent call - No streaming")
        result_tool= await Runner.run(self.agent_tool, "what is 3.42552 multiply by 562.3213")
        print(result_tool.final_output)
        print("---------------------------------------------------------------------")
        print()

    async def run_tool_agent_streaming(self):
        print("***********Tool agent call - Streaming")
        response = Runner.run_streamed(
            starting_agent=self.agent_tool,
            input="what is 7.814 multiplied by 103.892?"
        )

        async for event in response.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseFunctionCallArgumentsDeltaEvent):
                    # this is streamed parameters for our tool call
                    print(event.data.delta, end="", flush=True)
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    # this is streamed final answer tokens
                    print(event.data.delta, end="", flush=True)
            elif event.type == "agent_updated_stream_event":
                # this tells us which agent is currently in use
                print(f"> Current Agent: {event.new_agent.name}")
            elif event.type == "run_item_stream_event":
                # these are events containing info that we'd typically
                # stream out to a user or some downstream process
                if event.name == "tool_called":
                    # this is the collection of our _full_ tool call after our tool tokens have all been streamed
                    print()
                    print(f"> Tool Called, name: {event.item.raw_item.name}")
                    print(f"> Tool Called, args: {event.item.raw_item.arguments}")
                elif event.name == "tool_output":
                    # this is the response from our tool execution
                    print(f"> Tool Output: {event.item.raw_item['output']}")

    async def coversational_agent(self):
        response = await Runner.run(
            starting_agent=self.agent_tool,
            input="can you remember number 89"
        )

        #first it says it can't remember stuff
        print(f"response earlier : {response.final_output}") 
        #store
        response.to_input_list()
        #response from memeory
        response_conversation = await Runner.run(
            starting_agent=self.agent_tool,
            input=response.to_input_list() + [
                {"role": "user", "content": "what is last number I said multiplied by 2?"}
            ]
        )
        print(f"New Response: {response_conversation.final_output}")



async def main():
    demo = AgentDemo(agent, agent_tool)
    await demo.coversational_agent()

if __name__ == "__main__":
    asyncio.run(main())