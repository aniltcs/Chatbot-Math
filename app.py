import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from groq import APIError
import math

## Set upi the Stramlit app
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Using LLAMA model")

groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")

# -----------------------------
# Utility function to extract final content from agent
# -----------------------------
def extract_final_content(agent_result: dict) -> str:
    messages = agent_result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, (AIMessage, ToolMessage)) and msg.content:
            return msg.content
    return ""

if not groq_api_key:
    st.info("Please add your Groq APPI key to continue")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", streaming=True)


## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
# Wikipedia search tool
@tool("wikipedia_tool", return_direct=True)
def wikipedia_tool(query: str) -> str:
    """A tool for searching the Internet to find the vatious information on the topics mentioned."""
    return wikipedia_wrapper.run(query)

## Initializa the MAth tool

allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
allowed_names.update({"abs": abs, "round": round})

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Safely evaluates a math expression."""
    try:
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question
"""

## initialize the agents

assistant_agent=create_agent(
    llm,
    tools=[wikipedia_tool,calculator],
    system_prompt =prompt
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

tools=[wikipedia_tool,calculator]

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            response = assistant_agent.invoke(
                {
                    "messages": [{"role": "user", "content": question}],
                    "tools": tools
                },
                config={
                    "recursion_limit": 50,
                }
            )
            final_response = extract_final_content(response)
            st.session_state.messages.append({'role':'assistant',"content":final_response})
            st.write('### Response:')
            st.success(final_response)

    else:
        st.warning("Please enter the question")









