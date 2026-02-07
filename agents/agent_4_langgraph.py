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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic import hub
from typing import Dict, TypedDict

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
#research_agent = create_react_agent(model, [search], hub.pull("hwchase17/react"))
research_agent = create_agent(model=model, tools=[search],
                     system_prompt="You are a helpful agent that uses tools to answer questions accurately.")
research_node = ToolNode(tools=[search])

#calc_agent = create_react_agent(model, [profit_calc], hub.pull("hwchase17/react"))
calc_agent = create_agent(model=model, tools=[profit_calc],
                     system_prompt="You are a helpful agent that uses tools to answer questions accurately.")
calc_node = ToolNode(tools=[profit_calc])

# = create_react_agent(model, [summarize], hub.pull("hwchase17/react"))
summary_agent = create_agent(model=model, tools=[summarize],
                     system_prompt="You are a helpful agent that uses tools to answer questions accurately.")
summary_node = ToolNode(tools=[summarize])


# Extended State with iteration counter
class AgentState(TypedDict):
    messages: list
    next: str
    iterations: int

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

def supervisor(state: MessagesState) -> Dict[str, Literal["researcher", "calculator", "summarizer", "__end__"]]:
    messages = state["messages"]
    iters = state.get("iterations", 0)

    # LOOP PREVENTION: Max 5 iterations total
    if iters >= 5:
        return {"next": "FINALIZE", "iterations": iters}

    last_msg = messages[-1].content.lower()

    if any(word in last_msg for word in ["search", "latest", "current"]):
        return {"next": "researcher"}  # ← DICT RETURN
    elif any(word in last_msg for word in ["calculate", "profit", "cost"]):
        return {"next": "calculator"}
    elif "summarize" in last_msg:
        return {"next": "summarizer"}
    else:
        return {"next": "__end__"}

# Agent executor functions (decide tools vs FINAL ANSWER)
def research_agent(state: AgentState):
    messages = state["messages"]
    response = model.invoke(messages)
    # If no tool calls, it's final reasoning
    if not response.tool_calls:
        return {"messages": [response], "next": "FINALIZE"}
    return {"messages": [response]}

def calculator_agent(state: AgentState):
    messages = state["messages"]
    response = model.bind_tools([profit_calc]).invoke(messages)
    if not response.tool_calls:
        return {"messages": [response], "next": "FINALIZE"}
    return {"messages": [response]}

def summarizer_agent(state: AgentState):
    messages = state["messages"]
    response = model.bind_tools([summarize]).invoke(messages)
    if not response.tool_calls:
        return {"messages": [response], "next": "FINALIZE"}
    return {"messages": [response]}


# Multi-agent graph
workflow = StateGraph(MessagesState)

# Add specialist sub-graphs (simplified as nodes here; expand with agent.invoke in prod)
workflow.add_node("researcher", research_agent | research_node)
workflow.add_node("calculator", calc_agent | calc_node)
workflow.add_node("summarizer", summary_agent | summary_node)

# Add supervisor
workflow.add_node("supervisor", supervisor)

# Edges: supervisor → specialist → back to supervisor
#workflow.add_conditional_edges("supervisor", lambda s: s, {"researcher": "researcher", "calculator": "calculator", "summarizer": "summarizer", END: END})

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],  # Extract from dict
    {
        "researcher": "researcher",
        "calculator": "calculator",
        "summarizer": "summarizer",
        #"__end__": END
        "FINALIZE": END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("calculator", "supervisor")
workflow.add_edge("summarizer", "supervisor")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Multi-turn demo
config = {"configurable": {"thread_id": "multi-agent-session"}}
inputs = {"messages": [HumanMessage(content="Research LangGraph, calculate profit for $1M rev/$700k cost, then summarize.")]}
#for chunk in app.stream(inputs, config, stream_mode="values"):
#    chunk["messages"][-1].pretty_print()

for i, chunk in enumerate(app.stream(inputs, config, stream_mode="values")):
    if "messages" in chunk:
        print(f"--- Turn {i+1} ---")
        print(chunk["messages"][-1].content)
