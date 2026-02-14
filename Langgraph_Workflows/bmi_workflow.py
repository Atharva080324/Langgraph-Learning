from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float


# 1️⃣ Define function FIRST
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]

    bmi = weight / (height ** 2)
    state["bmi"] = round(bmi, 2)
    return state

def label_bmi(state:BMIState) -> BMIState:

    bmi = state["bmi"]

    if bmi < 18.5:
        state['category'] = "Underweight"
    elif 18.5 <= bmi <= 25:
        state['category'] = "Normal"
    elif 25 <= bmi <=30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"
    
    return state


# 2️⃣ Create graph AFTER function
graph = StateGraph(BMIState)

graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi",label_bmi)
graph.add_edge(START, "calculate_bmi")
graph.add_edge('calculate_bmi','label_bmi')
graph.add_edge("label_bmi", END)

workflow = graph.compile()

# 3️⃣ Run
initial_state =  {"weight": 70, "height": 1.75}
final_state = workflow.invoke(initial_state)

print(final_state)
png_bytes = workflow.get_graph().draw_mermaid_png()

with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as workflow_graph.png")

