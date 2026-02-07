# file: agent_3_memory.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain import hub
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

search = DuckDuckGoSearchRun()

@tool
def summarize_notes(topic: str) -> str:
    """Summarize key notes on a topic from memory."""
    return f"Notes on {topic}: Use tools, reason step-by-step, verify facts."

tools = [search, summarize_notes]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(model, tools, prompt)
memory = MemorySaver()  # Persists state across invocations
agent_executor = agent.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}  # Session ID

# First interaction
input1 = {"messages": [("user", "What is LangGraph? Search if needed.")]}
response1 = agent_executor.invoke(input1, config)
print("Turn 1:", response1["messages"][-1].content)

# Follow-up (remembers prior!)
input2 = {"messages": [("user", "Summarize notes from before.")]}
response2 = agent_executor.invoke(input2, config)
print("Turn 2:", response2["messages"][-1].content)
