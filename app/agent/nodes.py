# the node functions that make up the graph, each node takes agent state and returns a partial update
# then langgraph merges that update back into the state before running whichever node comes next
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.agent.state import AgentState
from app.agent.tools import create_order, retrieve_product_info, retrieve_support_info

_llm = None


def get_llm():
    # lazy-singleton pattern 
    # build the LLM client once, reuse it across every node call instead
    # of recreating it on every single message.
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.environ.get("GROQ_API_KEY"))
    return _llm


def classify_intent(state: AgentState) -> dict:
    """First node in the graph. Asks the LLM a narrow yes/no-style
    question — sales or customer_service — and stores the answer in
    state['intent'], which the conditional edge reads next."""
    llm = get_llm()
    last_message = state["messages"][-1].content

    system_prompt = (
        "Classify the customer's message as exactly one word: "
        "'sales' if they're asking about products, prices, recommendations, "
        "or want to buy something; 'customer_service' if they're asking "
        "about shipping, returns, policies, or general support. "
        "Respond with only one of these two words, nothing else."
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
    intent = response.content.strip().lower()

    # Defensive fallback: if the LLM ever responds with something other
    # than exactly one of the two expected words, defaults to the safer,
    # non-purchasing path.
    if intent not in ("sales", "customer_service"):
        intent = "customer_service"

    return {"intent": intent}


def route_by_intent(state: AgentState) -> str:
    """The conditional edge function, called after
    classify_intent and uses its return value (a node name) to decide
    which node runs next."""
    return state["intent"]


def _execute_tool_calls(ai_message, tools_list):
    """Shared by sales_node and customer_service_node. The LLM only
    decides which tool to call and with what arguments, it doesn't run
    any code itself. This function is what actually executes the real
    Python function and packages the result back into a ToolMessage the
    LLM can read on its next turn."""
    tool_lookup = {t.name: t for t in tools_list}
    tool_messages = []
    retrieved_context = None
    tool_result = None

    for call in ai_message.tool_calls:
        tool_fn = tool_lookup[call["name"]]
        result = tool_fn.invoke(call["args"])

        if call["name"] == "create_order":
            tool_result = result
        else:
            retrieved_context = result

        # feed the tool's result back to the LLM
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return tool_messages, retrieved_context, tool_result


def sales_node(state: AgentState) -> dict:
    """Handles sales conversation. Bound to two tools: retrieve_product_info
    (product questions) and create_order (the real business action), the
    LLM decides which, if either, to call based on the conversation."""
    tools_list = [retrieve_product_info, create_order]
    # bind the tools to the LLM so it can call them by name in its reasoning
    llm = get_llm().bind_tools(tools_list)

    system_prompt = (
        "You are a sales assistant for an e-commerce store. Help the "
        "customer find products and answer questions about price and "
        "availability using retrieve_product_info. Only call create_order "
        "after the customer has clearly confirmed what they want to buy. "
        f"When calling create_order, always use customer_id={state['customer_id']}."
    )

    # Sends system message and conversation history to the LLM
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    ai_message = llm.invoke(messages)

    if not ai_message.tool_calls:
        # tool_calls is empty if the LLM didn't decide to call any tools, in which case we just return its message as-is
        return {"messages": [ai_message]}

    tool_messages, retrieved_context, tool_result = _execute_tool_calls(ai_message, tools_list)

    # Ask the LLM again, now with the tool's real result included, so it
    # can turn raw data (retrieved text, or an order confirmation dict)
    # into an actual conversational reply.
    follow_up = messages + [ai_message] + tool_messages
    final_message = llm.invoke(follow_up)

    update = {"messages": [ai_message] + tool_messages + [final_message]}
    if retrieved_context is not None:
        update["retrieved_context"] = retrieved_context
    if tool_result is not None:
        update["tool_result"] = tool_result
    return update


def customer_service_node(state: AgentState) -> dict:
    """Handles customer-service conversation. Bound to only retrieve_support_info"""
    tools_list = [retrieve_support_info]
    llm = get_llm().bind_tools(tools_list)

    system_prompt = (
        "You are a customer service assistant for an e-commerce store. "
        "Answer questions about shipping, returns, and policies using "
        "retrieve_support_info to find accurate information before answering."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    ai_message = llm.invoke(messages)

    if not ai_message.tool_calls:
        return {"messages": [ai_message]}

    tool_messages, retrieved_context, _ = _execute_tool_calls(ai_message, tools_list)

    follow_up = messages + [ai_message] + tool_messages
    final_message = llm.invoke(follow_up)

    update = {"messages": [ai_message] + tool_messages + [final_message]}
    if retrieved_context is not None:
        update["retrieved_context"] = retrieved_context
    return update


def format_response(state: AgentState) -> dict:
    """Last node before END. Pulls the agent's finished reply out of the
    conversation history into state['response'] , giving the Flask chat
    route one single, predictable field to read the final answer from,
    regardless of which branch (sales or customer service) produced it."""
    last_message = state["messages"][-1]
    return {"response": last_message.content}