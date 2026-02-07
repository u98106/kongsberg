# file: agent_1_basic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain import hub

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # Mock; replace with Tavily/DuckDuckGo in prod
    return f"Search results for '{query}': Agentic AI uses tools autonomously. LangChain enables this."

tools = [web_search]
prompt = hub.pull("hwchase17/react")  # Standard ReAct prompt

agent = create_react_agent(model, tools, prompt)
agent_executor = agent.compile()

response = agent_executor.invoke({"messages": [("user", "What is agentic AI?")]})
print(response["messages"][-1].content)
