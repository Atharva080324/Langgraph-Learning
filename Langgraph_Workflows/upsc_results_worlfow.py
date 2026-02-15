from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from pydantic import BaseModel,Field
import os
import operator

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Feedback on the evaluation results")
    score: int = Field(description="Score of the evaluation on a scale of 1 to 10",ge=0,le=10
)
    
structured_model = model.with_structured_output(EvaluationSchema)

essay = """Climate change is one of the most pressing issues facing our planet today. It refers to the long-term alteration of temperature and typical weather patterns in a place. Climate change could refer to a particular location or the planet as a whole. The primary cause of climate change is human activity, particularly the burning of fossil fuels which releases greenhouse gases into the atmosphere. These gases trap heat and cause the Earth's temperature to rise, leading to a variety of environmental impacts such as rising sea levels, more frequent and severe weather events, and loss of biodiversity. To combat climate change, it is crucial for individuals, governments, and organizations to take action by reducing carbon emissions, transitioning to renewable energy sources, and implementing sustainable practices in various sectors."""

prompt = f"Evaluate the following essay on the topic of climate change and provide feedback along with a score out of 10: -{essay}"

evaluation = structured_model.invoke(prompt)

class UPSCState(TypedDict):
    essay:str
    language_feedback:str
    clarity_feedback:str
    depth_feedback:str
    overall_feedback:str
    individual_scores:Annotated[list[int],operator.add]
    average_score:float

def Language(state:UPSCState):
    prompt = f"Evaluate the following essay on language quality and provide feedback along with a score out of 10: -{state['essay']}"
    evaluation = structured_model.invoke(prompt)
    return {"language_feedback": evaluation.feedback, "individual_scores": [evaluation.score]}

def Clarity_Of_Thought(state:UPSCState):
    prompt = f"Evaluate the following essay on clarity of thought and provide feedback along with a score out of 10: -{state['essay']}"
    evaluation = structured_model.invoke(prompt)
    return {"clarity_feedback": evaluation.feedback, "individual_scores": [evaluation.score]}

def Depth_Of_Analysis(state:UPSCState):
    prompt = f"Evaluate the following essay on depth of analysis and provide feedback along with a score out of 10: -{state['essay']}"
    evaluation = structured_model.invoke(prompt)
    return {"depth_feedback": evaluation.feedback, "individual_scores": [evaluation.score]}

def Overall_Evaluation(state:UPSCState):
    prompt = f"Based on the following feedback and scores, provide an overall evaluation of the essay along with an average score out of 10: - Language Feedback: {state['language_feedback']} with score {state['individual_scores'][0]}, Clarity of Thought Feedback: {state['clarity_feedback']} with score {state['individual_scores'][1]}, Depth of Analysis Feedback: {state['depth_feedback']} with score {state['individual_scores'][2]}"
    evaluation = model.invoke(prompt)
    state["overall_feedback"] = evaluation.content
    state["average_score"] = sum(state["individual_scores"])/len(state["individual_scores"])
    return {"overall_feedback": state["overall_feedback"], "average_score": state["average_score"]}

graph = StateGraph(UPSCState)

graph.add_node("Clarity_Of_Thought",Clarity_Of_Thought)
graph.add_node("Depth_Of_Analysis",Depth_Of_Analysis)
graph.add_node("Language",Language)
graph.add_node("Overall_Evaluation",Overall_Evaluation)

graph.add_edge(START, "Clarity_Of_Thought")
graph.add_edge(START, "Depth_Of_Analysis")
graph.add_edge(START, "Language")
graph.add_edge("Clarity_Of_Thought", "Overall_Evaluation")
graph.add_edge("Depth_Of_Analysis", "Overall_Evaluation")
graph.add_edge("Language", "Overall_Evaluation")
graph.add_edge("Overall_Evaluation", END)

workflowgraph = graph.compile()

essay = """Climate change is one of the most pressing issues facing our planet today. It refers to the long-term alteration of temperature and typical weather patterns in a place. Climate change could refer to a particular location or the planet as a whole. The primary cause of climate change is human activity, particularly the burning of fossil fuels which releases greenhouse gases into the atmosphere. These gases trap heat and cause the Earth's temperature to rise, leading to a variety of environmental impacts such as rising sea levels, more frequent and severe weather events, and loss of biodiversity. To combat climate change, it is crucial for individuals, governments, and organizations to take action by reducing carbon emissions, transitioning to renewable energy sources, and implementing sustainable practices in various sectors.  """

initial_state = {
    "essay": essay
}

final_state = workflowgraph.invoke(initial_state)
print("Final State:", final_state)




