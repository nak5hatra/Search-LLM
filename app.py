import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun, Tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler

##  Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=11, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

##  Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

serper_api_wrapper = GoogleSerperAPIWrapper()


search = Tool(
    name="Google Search",
    func=serper_api_wrapper.run,
    description="useful for when you need to search for real-time information from Google."
)

st.title("LangChain App - Chat and Search")

# """
# In this app, I'm using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.

# """

##  Sidebar for Settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
serper_api_key = st.sidebar.text_input("Enter your serper API Key:", type="password")

serper_api_wrapper = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


search = Tool(
    name="Google Search",
    func=serper_api_wrapper.run,
    description="useful for when you need to search for real-time information from Google."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content":"Hi! I'm a Chatbot who can search the web. How can I help you today?"
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="Ask me Anything! 🤗"):
    st.session_state.messages.append({"role": "user","content":prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    
    tools = [arxiv, search, wikipedia]
    search_agent = initialize_agent(tools=tools,llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content": response 
            }
        )
        st.write(response)
    
