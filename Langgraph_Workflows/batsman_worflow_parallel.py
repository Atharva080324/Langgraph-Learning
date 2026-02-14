from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class BatsmanState(TypedDict):
    runs:int
    balls:int
    fours:int
    sixes:int
    sr:float
    bpb:float
    boundary_percentage:float
    summary:str


def calculate_sr(state:BatsmanState)->BatsmanState:
    runs = state["runs"]
    balls = state["balls"]
    state["sr"] = (runs/balls)*100 if balls > 0 else 0
    return {"sr": state["sr"]}

def calculate_bpb(state:BatsmanState)->BatsmanState:
    runs = state["runs"]
    balls = state["balls"]
    state["bpb"] = runs/balls if balls > 0 else 0
    return {"bpb": state["bpb"]}

def calculate_boundary_percentage(state:BatsmanState)->BatsmanState:
    runs = state["runs"]
    fours = state["fours"]
    sixes = state["sixes"]
    boundaries_runs = (fours*4) + (sixes*6)
    state["boundary_percentage"] = (boundaries_runs/runs)*100 if runs > 0 else 0
    return {"boundary_percentage": state["boundary_percentage"]}

def final_summary(state:BatsmanState)->BatsmanState:
    runs = state["runs"]
    balls = state["balls"]
    fours = state["fours"]
    sixes = state["sixes"]
    sr = state["sr"]
    bpb = state["bpb"]
    boundary_percentage = state["boundary_percentage"]

    prompt = f"Provide a summary of a batsman's performance with the following details - runs scored: {runs}, balls faced: {balls}, number of fours: {fours}, number of sixes: {sixes}, strike rate: {sr}, runs per ball: {bpb} and boundary percentage: {boundary_percentage}. The summary should be concise and highlight the key aspects of the performance."

    summary = model.invoke(prompt).content
    state["summary"] = summary
    return state



graph = StateGraph(BatsmanState)

graph.add_node("calculate_sr",calculate_sr)
graph.add_node("calculate_bpb",calculate_bpb)
graph.add_node("calculate_boundary_percentage",calculate_boundary_percentage)
graph.add_node("final_summary",final_summary)

graph.add_edge(START,"calculate_sr")
graph.add_edge(START,"calculate_bpb")
graph.add_edge(START,"calculate_boundary_percentage")
graph.add_edge("calculate_sr","final_summary")
graph.add_edge("calculate_bpb","final_summary")
graph.add_edge("calculate_boundary_percentage","final_summary")

graph.add_edge("final_summary",END)

workflow = graph.compile()

initial_state = {"runs":120,
                "balls":80,
                "fours":10,
                "sixes":2
}  

final_state = workflow.invoke(initial_state)

print(final_state['summary'])

png_bytes = workflow.get_graph().draw_mermaid_png()

with open("batsman_workflow_graph.png", "wb") as f:
    f.write(png_bytes)





