# file: agent_4_langgraph.py
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain import hub

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search = DuckDuckGoSearchRun()

# Shared tools
@tool
def profit_calc(revenue: float, cost: float) -> float:
    """Calculate business profit."""
    return revenue - cost

@tool
def summarize(topic: str) -> str:
    """Generate concise summary."""
    return f"Summary for {topic}: Autonomous, tool-using AI systems."

tools = [search, profit_calc, summarize]

# Specialist agents (sub-graphs)
research_agent = create_react_agent(model, [search], hub.pull("hwchase17/react"))
research_node = ToolNode(tools=[search])

calc_agent = create_react_agent(model, [profit_calc], hub.pull("hwchase17/react"))
calc_node = ToolNode(tools=[profit_calc])

summary_agent = create_react_agent(model, [summarize], hub.pull("hwchase17/react"))
summary_node = ToolNode(tools=[summarize])

# Supervisor decides routing
def supervisor(state: MessagesState):
    messages = state["messages"]
    last_msg = messages[-1].content.lower()
    
    if any(word in last_msg for word in ["search", "latest", "current"]):
        return "researcher"
    elif any(word in last_msg for word in ["calculate", "profit", "cost"]):
        return "calculator"
    elif "summarize" in last_msg:
        return "summarizer"
    else:
        return END

# Multi-agent graph
workflow = StateGraph(MessagesState)

# Add supervisor
workflow.add_node("supervisor", supervisor)

# Add specialist sub-graphs (simplified as nodes here; expand with agent.invoke in prod)
workflow.add_node("researcher", research_agent | research_node)
workflow.add_node("calculator", calc_agent | calc_node)
workflow.add_node("summarizer", summary_agent | summary_node)

# Edges: supervisor → specialist → back to supervisor
workflow.add_conditional_edges("supervisor", lambda s: s, {"researcher": "researcher", "calculator": "calculator", "summarizer": "summarizer", END: END})
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("calculator", "supervisor")
workflow.add_edge("summarizer", "supervisor")

workflow.set_entry_point("supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Multi-turn demo
config = {"configurable": {"thread_id": "multi-agent-session"}}
inputs = {"messages": [HumanMessage(content="Research LangGraph, calculate profit for $1M rev/$700k cost, then summarize.")]}
for chunk in app.stream(inputs, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
