import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class BlogState(TypedDict):
    topic : str
    outline : str
    results : str
    score : int 

graph = StateGraph(BlogState)

def create_outline(state:BlogState)->BlogState:
    topic = state["topic"]

    prompt = f"Generate an outline for a blog on a topic -{topic}"
    outline = model.invoke(prompt).content
    state["outline"] = outline
    return state

def write_blog(state:BlogState)->BlogState:
    topic = state["topic"]
    outline = state["outline"]
    prompt = f"Write a blog on the topic - {topic} with the following outline - {outline}"

    blog = model.invoke(prompt).content
    state["results"] = blog
    return state

def evaluation(state:BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    blog = state["results"]

    prompt = f"Evaluate the blog written on the topic - {topic} with the following outline - {outline} and the blog content - {blog}. Provide a score out of 10 only,return an integer value only."
    score = model.invoke(prompt).content
    state["score"] = int(score)
    return state

graph.add_node("create_outline",create_outline)
graph.add_node("write_blog",write_blog)
graph.add_node("evaluation",evaluation)

graph.add_edge(START,"create_outline")
graph.add_edge("create_outline","write_blog")
graph.add_edge("write_blog","evaluation")
graph.add_edge("evaluation",END)

workflow = graph.compile()

initial_state = {"topic":"Rise of AI in India"}
final_state = workflow.invoke(initial_state)

print(final_state)
