# file: agent_2_multitool.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun  # Built-in
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain import hub

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

search = DuckDuckGoSearchRun()  # Real search!

@tool
def calculate_profit(revenue: float, cost: float) -> float:
    """Calculate profit from revenue and cost."""
    return revenue - cost

tools = [search, calculate_profit]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(model, tools, prompt)
agent_executor = agent.compile()

response = agent_executor.invoke({
    "messages": [("user", "Search latest LangChain version. If revenue $500k and cost $350k, what's profit?")]
})
print(response["messages"][-1].content)
