import os

from typing import Any
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub


from pydantic import BaseModel

from langserve import add_routes

from operator import itemgetter

from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# vectorstore = FAISS.from_texts(
#     ["cats like fish", "dogs like sticks"], embedding=HuggingFaceEmbeddings()
# )

# retriever = vectorstore.as_retriever()


# @tool
# def get_eugene_thoughts(query: str) -> list:
#     """Returns Eugene's thoughts on a topic."""
#     return retriever.get_relevant_documents(query)


tools = [TavilySearchResults(max_results=1)]


prompt = hub.pull("hwchase17/react")


llm = ChatGroq(model="llama3-8b-8192", temperature=0, streaming=True)

# llm_with_tools = llm.bind_tools(tools)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke({"input": "Onde fica a biofy"})