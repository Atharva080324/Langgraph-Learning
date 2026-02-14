from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class MemoryState(TypedDict):
    question: str
    answer: str

def LLMcall(state: MemoryState) -> MemoryState:
    question = state["question"]                  # FIX
    answer = model.invoke(question).content
    state["answer"] = answer                      # FIX
    return state                                  # FIX

graph = StateGraph(MemoryState)

graph.add_node("LLMcall", LLMcall)
graph.add_edge(START, "LLMcall")
graph.add_edge("LLMcall", END)

workflow = graph.compile()

initial_state = {"question": "What is the capital of India?"}
final_state = workflow.invoke(initial_state)

print(final_state['answer'])
png_bytes = workflow.get_graph().draw_mermaid_png()

with open("llm_workflow_graph.png", "wb") as f:
    f.write(png_bytes)
