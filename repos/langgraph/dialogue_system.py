from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
import asyncio

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_query: str
    search_results: str
    final_answer: str
    step: str

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature=0.7
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def understand_query_node(state: SearchState) -> dict:
    user_message = state["messages"][-1].content
    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""
    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    search_query = user_message
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()

    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }

def tavily_search_node(state: SearchState) -> dict:
    search_query = state["search_query"]
    try:
        print ("searching")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        search_results = response

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="search done, organizing...")]
        }

    except Exception as e:
        return {
            "search_results": f"search fail, {e}",
            "step": "search_failed",
            "messages": [AIMessage(content="search fail with some issues...")]
        }

def generate_answer_node(state: SearchState) -> dict:
    if state["step"] == "search_failed":
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：\n{state['search_results']}
请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

def create_search_assistant():
    workflow = StateGraph(SearchState)
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app

async def main():
    app = create_search_assistant()

    session_count = 0

    while True:
        user_input = input("what do you want to know?").strip()

        if user_input.lower() in["quit"]:
            break

        if not user_input:
            continue

        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }

        try:
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print (f"understanding stage: {latest_message.content}")
                            elif node_name == "search":
                                print (f"searching stage: {latest_message.content}")
                            elif node_name == "answer":
                                print (f"final answer: {latest_message.content}")
        except Exception as e:
            print(f"error!!!!! {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())





