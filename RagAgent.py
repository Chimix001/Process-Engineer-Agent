#importing depencies

from typing import TypedDict, Sequence, Annotated
from langchain_core.messages import (
     BaseMessage, HumanMessage,
       AIMessage, SystemMessage )
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

#loading the environment variables
load_dotenv()

#initializing the LLM
llm = ChatGoogleGenerativeAI( model="gemini-3.5-flash", temperature=0 )

#prompt

template = """ Answer the question based on the following context and chat history. 
Especially take the latest question into consideration.

Chat history: 
{history}

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

#Agent State

class AgentState(TypedDict):
    """The state of the agent. 
    LangGraph keeps track of these values while the workflow runs."""
    messages: Annotated[
        Sequence[BaseMessage],
        add_messages
    ]
    on_topic: bool

#question classifer

class QuestionClassification(BaseModel):
    is_chemical_related: bool = Field(
        description= (
        "True if the question is related to chemical engineering," 
        "process engineering, castor oil production, castor oil extraction," 
        "oil yield calculations, mass balance, unit conversion,"
        "castor oil properties, or biodiesel.")
    )

# Create a structured-output version of the LLM
classifer_llm = llm.with_structured_output(
    QuestionClassification,
      method="json_mode" )

#question classifier 
def question_classifier(state: AgentState):
    print("Entering question classifier")

    #get the latest user question
    question = state["messages"][-1].content

    result = classifer_llm.invoke(
        f"""
        Determine whether this question is related to a 
        chemical process engineering assistant.

        The assistant can help with:

        - Castor oil extraction
        - Materials and methods used to produce castor oil
        - Physicochemical properties of castor oil
        - Determination of density
        - Determination of kinematic viscosity
        - Determination of moisture content
        - Determination of flash point
        - Determination of fire point
        - Determination of pour point
        - Determination of saponification value
        - Esterification experiments
        - Transesterification experiments
        - Characterization of castor oil
        - Characterization of biodiesel
        - Mass balance
        - Oil yield calculations
        - Unit conversion
        - Chemical process engineering

        Question:
        {question}
        Return true if the question is related to the topics above.
        Return false if the question is unrelated.
        Respond with JSON only.

        The JSON must have exactly this structure:

        {{
            "is_chemical_related": true
              }} 
              """
                )
    print(
        f"chemical related: {result.is_chemical_related}"
    )

    return {
        "on_topic: result.is_chemical_related"
    }

#Topic ROuter
def topic_router(state: AgentState):
    print("Entering Topic Router...")
    if state["on_topic"]:
        print("Question relating to chemical Engineering")
        return "retriever"
    print("Question off_topic")

    return "off_topic"

#off_topic response

def off_topic_response(state: AgentState):
    print("Entering off topic response")

    return{
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry! I can only help with "
                    "chemical engineering, process engineering, "
                    "castor oil production, and castor oil extraction."
                )
            )
        
        ]
    }

#cannot answer
def cannot_answer(state: AgentState):
    print("Entering cannot_answer...")
    return {
        "messages": [ AIMessage(
            content=(
                "I'm sorry, but I cannot find " "the information you're looking for." ) ) ] }

#Memory
checkpointer = MemorySaver()

#build the LangGraph

builder = StateGraph(AgentState)

#Add Nodes

builder.add_node("classifier", question_classifier)
builder.add_node("off_topic", off_topic_response)
builder.add_node("cannot_answer", cannot_answer)

#Edges
builder.add_edge(START, "classifier")

#classifier - Topic Router

builder.add_conditional_edges("classifier", topic_router,
                              {"retriever": "cannot_answer",
                               "off_topic": "off_topic" })

#off_topic - END

builder.add_edge("off_topic", END)

# CANNOT_ANSWER → END 
builder.add_edge( "cannot_answer", END )

#COMPILE GRAPH

graph = builder.compile(checkpointer=checkpointer)

#testing the graph

if __name__ == "__name__":
    quesstion = input("\nAsk a question")

    result = graph.invoke(
        {
            "messages":[
                HumanMessage(content=quesstion)
                ],
                "on_topic": False
        },
        config={
            "configurable":{
                "thread_id": "1"
            }
        }
    )

    print("\nfinal response:")

    for message in result["messages"]:
        print(message.content)