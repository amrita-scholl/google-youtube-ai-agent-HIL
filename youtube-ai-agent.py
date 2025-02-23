import os
import langgraph
from langgraph.graph import StateGraph
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

# Define AI Model (ChatGPT)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Define State Schema
class YouTubeState(TypedDict):
    query: str
    search_results: Optional[str]
    best_answer: Optional[str]

# Define Nodes
def youtube_search_node(state: YouTubeState) -> YouTubeState:
    """Fetches top YouTube video results using SerpAPI manually."""
    from serpapi import GoogleSearch

    query = state["query"]
    params = {
        "engine": "youtube",
        "search_query": query,
        "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    
    video_results = results.get("video_results", [])
    
    if video_results:
        formatted_results = "\n".join([f"{i+1}. {v['title']} | {v['link']}" for i, v in enumerate(video_results[:5])])
    else:
        formatted_results = "No videos found."

    return {"search_results": formatted_results}

def ai_summary_node(state: YouTubeState) -> YouTubeState:
    """Uses AI to summarize the best video description."""
    if state["search_results"]:
        summary = llm.predict(f"Based on the following YouTube search results, which video best answers the query: {state['query']}? Provide a brief summary:\n{state['search_results']}")
        return {"best_answer": summary}
    return {"best_answer": "No relevant YouTube videos found."}

# Define LangGraph Workflow
workflow = StateGraph(YouTubeState)

workflow.add_node("youtube_search", youtube_search_node)
workflow.add_node("ai_summary", ai_summary_node)

workflow.set_entry_point("youtube_search")
workflow.add_edge("youtube_search", "ai_summary")

graph = workflow.compile()

# API Endpoint to Trigger YouTube AI Agent
@app.post("/search")
async def search_agent(data: dict):
    query = data.get("query", "")
    output = graph.invoke({"query": query})
    return {
        "search_results": output.get("search_results", "No results found"),
        "best_answer": output.get("best_answer", "No summary available")
    }

# Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
