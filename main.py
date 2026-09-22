from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from RagAgent import graph
from RagAgent import start_mcp, stop_mcp


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting MCP connection...")

    await start_mcp()

    print("MCP connection ready.")

    yield

    print("Stopping MCP connection...")

    await stop_mcp()


app = FastAPI(
    title="Castor Oil AI Agent",
    version="0.1.0",
    lifespan=lifespan
)


class Question(BaseModel):
    message: str


@app.get("/")
def home():

    return {
        "message": "Castor Oil AI Agent is running"
    }


@app.post("/ask")
async def ask_agent(request: Question):

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=request.message)
            ],
            "on_topic": False,
            "context": ""
        },
        config={
            "configurable": {
                "thread_id": "fastapi-user"
            }
        }
    )

    return {
        "question": request.message,
        "answer": result["messages"][-1].content
    }