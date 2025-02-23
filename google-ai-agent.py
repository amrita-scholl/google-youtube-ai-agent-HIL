import os
import langgraph
from langgraph.graph import StateGraph  # ✅ Correct Import
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from fastapi import FastAPI
from dotenv import load_dotenv
from typing import TypedDict, Optional

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Set up environment variables
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Define AI Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# ✅ Define State Schema for LangGraph
class SearchState(TypedDict):
    query: str
    approved_query: Optional[str]
    search_results: Optional[str]
    ai_summary: Optional[str]

# ✅ Set up Google Search Tool
google_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)

# ✅ Define Nodes for LangGraph
def human_verification(state: SearchState) -> SearchState:
    """Simulates human verification of the query."""
    print(f"AI Suggested: {state['query']}")
    decision = input("Do you approve this search? (yes/no): ").strip().lower()
    return {"approved_query": state["query"] if decision == "yes" else None}

def google_search_node(state: SearchState) -> SearchState:
    """Runs Google search if the query is approved."""
    query = state.get("approved_query")
    if query:
        results = google_search.run(query)
        return {"search_results": results}
    return {"search_results": "Query was rejected by human."}

def generate_summary(state: SearchState) -> SearchState:  # ✅ Renamed node
    """Summarizes search results using AI."""
    search_results = state.get("search_results")
    if search_results and search_results != "Query was rejected by human.":
        summary = llm.predict(f"Summarize the following search results:\n{search_results}")
        return {"ai_summary": summary}
    return {"ai_summary": "No valid search results to summarize."}

# ✅ Define LangGraph Workflow
workflow = StateGraph(SearchState)  # ✅ Now we define the schema!

# ✅ Add nodes
workflow.add_node("human_loop", human_verification)
workflow.add_node("google_search", google_search_node)
workflow.add_node("generate_summary", generate_summary)  # ✅ Renamed node

# ✅ Define workflow structure
workflow.set_entry_point("human_loop")
workflow.add_edge("human_loop", "google_search")
workflow.add_edge("google_search", "generate_summary")  # ✅ Renamed node

# ✅ Compile the graph
graph = workflow.compile()

# ✅ API Endpoint to Trigger AI Agent
@app.post("/search")
async def search_agent(data: dict):
    query = data.get("query", "")
    output = graph.invoke({"query": query})  # ✅ Ensure input matches schema
    return {
        "serpapi_response": output.get("search_results", "No search results"),
        "chatgpt_summary": output.get("ai_summary", "No AI summary")
    }

# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
