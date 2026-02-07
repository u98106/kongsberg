# file: agent_3_memory.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_classic import hub
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

#agent = create_react_agent(model, tools, prompt)
agent = create_agent(model=model, tools=tools,
                     system_prompt="You are a helpful agent that uses tools to answer questions accurately.")
memory = MemorySaver()  # Persists state across invocations
#agent_executor = agent.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}  # Session ID

# First interaction
input1 = {"messages": [("user", "What is LangGraph? Search if needed.")]}
for chunk in agent.stream(input1, stream_mode="updates"):
    if "model" in chunk:
        print('Turn 1:', chunk["model"]["messages"][-1].content)

# Follow-up (remembers prior!)
input2 = {"messages": [("user", "Summarize notes from before.")]}
for chunk in agent.stream(input2, stream_mode="updates"):
    if "model" in chunk:
        print('Turn 2:', chunk["model"]["messages"][-1].content)
