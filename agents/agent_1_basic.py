# file: agent_1_basic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_classic import hub

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # Mock; replace with Tavily/DuckDuckGo in prod
    return f"Search results for '{query}': Agentic AI uses tools autonomously. LangChain enables this."

tools = [web_search]
prompt = hub.pull("hwchase17/react")  # Standard ReAct prompt

agent = create_agent(model=model, tools=tools,
                     system_prompt="You are a helpful agent that uses tools to answer questions accurately.")
#agent_executor = agent.compile()

#response = agent_executor.invoke({"messages": [("user", "What is agentic AI?")]})
#print(response["messages"][-1].content)

inputs = {"messages": [("user", "What is agentic AI?")]}
for chunk in agent.stream(inputs, stream_mode="updates"):
    #if "agent" in chunk:
    #    print(chunk["agent"]["messages"][-1].content)
    if "model" in chunk:
        print(chunk["model"]["messages"][-1].content)
    #print(chunk)
