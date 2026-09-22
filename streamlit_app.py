import asyncio
import streamlit as st

from langchain_core.messages import HumanMessage

from RagAgent import graph, start_mcp, stop_mcp


st.set_page_config(
    page_title="Castor Oil AI Assistant",
    page_icon="🌱",
    layout="centered"
)


# --------------------------------------------------
# MCP CONNECTION
# --------------------------------------------------

@st.cache_resource
def get_mcp_connection():

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(start_mcp())

    return loop


# Start MCP
loop = get_mcp_connection()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🌱 Castor Oil AI")

    st.write(
        """
        AI assistant for:

        - Castor oil extraction
        - Castor oil properties
        - Oil yield calculations
        - Moisture content
        - Specific gravity
        - Kinematic viscosity
        - Unit conversions
        - Chemical process engineering
        """
    )

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.title("🌱 Castor Oil AI Assistant")

st.caption(
    "RAG + LangGraph + MCP + Groq"
)

st.write(
    "Ask questions about castor oil and chemical process engineering."
)


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about castor oil..."
)


if question:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = loop.run_until_complete(
                    graph.ainvoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=question
                                )
                            ],
                            "on_topic": False,
                            "context": ""
                        },
                        config={
                            "configurable": {
                                "thread_id": "streamlit-user"
                            }
                        }
                    )
                )

                answer = result["messages"][-1].content

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )