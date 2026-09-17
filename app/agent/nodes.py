# the node functions that make up the graph, each node takes agent state and returns a partial update
# then langgraph merges that update back into the state before running whichever node comes next
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from app.agent.state import AgentState
from app.agent.tools import (
    add_to_cart,
    check_product_availability,
    create_order,
    retrieve_product_info,
    retrieve_support_info,
)

_llm = None

_SHARED_BEHAVIOR_RULES = (
    "Formatting: write short, natural, conversational sentences — never a "
    "markdown table, never raw pipe-delimited data, never a bulleted dump "
    "of every field a tool returned. Summarize like a helpful person "
    "would, not like you're printing a database row. "
    "Never mention a product's internal id number to the customer — it's "
    "for your own use when calling tools, not something a customer needs "
    "to see or hear. Refer to products by name. "
    "The same applies to the customer's own internal customer_id/account "
    "number — never say things like \"your account, which is customer #4\" "
    "or state that number at all; just say \"your account\" with no number. "
    "(An order's id, like \"Order #5,\" is different and fine to share — "
    "that's a normal order confirmation number, not an internal account id.) "
    "Stay on topic: you only help with this store — products, orders, "
    "shipping, returns, and policies. If asked something unrelated (general "
    "knowledge, other topics, anything outside the store), politely say "
    "that's outside what you can help with here and steer back to the "
    "store. "
    "Never reveal or name your internal tools or functions, even if asked "
    "directly what tools, functions, or capabilities you have — describe "
    "what you can help with in plain customer-friendly terms instead "
    "(e.g. \"I can help you find products and check availability,\" never "
    "\"I have a retrieve_product_info tool\")."
)

def get_llm():
    # lazy-singleton pattern 
    # build the LLM client once, reuse it across every node call instead
    # of recreating it on every single message.
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", 
                        api_key=os.environ.get("GROQ_API_KEY"),
                        temperature=0.1,)
    return _llm


def classify_intent(state: AgentState) -> dict:
    """First node in the graph. Asks the LLM a narrow yes/no-style
    question — sales or customer_service — and stores the answer in
    state['intent'], which the conditional edge reads next."""
    llm = get_llm()
    last_message = state["messages"][-1].content

    system_prompt = (
        "Classify the customer's LATEST message as exactly one word: "
        "'sales' if they're asking about products, prices, recommendations, "
        "or want to buy something; 'customer_service' if they're asking "
        "about shipping, returns, warranties, policies, or general support "
        "— even if the question also mentions a product category (like "
        "\"electronics\"), a question about warranty terms, return rules, "
        "or any store policy is customer_service, not sales, regardless of "
        "what product it's about. "
        "The latest message might be short and depend on earlier context "
        "in the conversation (e.g. a bare number answering a previous "
        "question about quantity) — use the full conversation to "
        "understand what it actually refers to, not just its own words. "
        "Respond with only one of the two words, nothing else."
    )
 

    # Full conversation history, not just the isolated last message — a
    # short reply like "1" is meaningless classified on its own; it only
    # makes sense in light of what was being discussed before it.
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
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

# before: each node called LLM exactly twice , not more
# now: each node can call LLM as many times as it needs to, until it either stops requesting tools or hits a safety cap of max_iterations.
# in case it needs more than one tool call
def _run_agent_loop(llm, messages: list, tools_list: list, max_iterations: int = 4) -> dict:
    """Repeatedly calls the LLM and executes whatever tools it requests,
    feeding results back in, until it responds without requesting another
    tool call, or until max_iterations is hit as a safety cap against a
    runaway loop. This replaces a fixed "call once, maybe run tools once,
    call once more" shape, which can't handle a question that genuinely
    needs two sequential tool calls, e.g. retrieve_product_info to find
    a product's id, then check_product_availability using that id, which
    can't both be known before the first tool's result comes back."""
    new_messages = []
    retrieved_context = None
    tool_result = None
 
    for _ in range(max_iterations):
        ai_message = llm.invoke(messages + new_messages)
        new_messages.append(ai_message)
 
        if not ai_message.tool_calls:
            break
 
        tool_messages, this_context, this_result = _execute_tool_calls(ai_message, tools_list)
        new_messages.extend(tool_messages)
        if this_context is not None:
            retrieved_context = this_context
        if this_result is not None:
            tool_result = this_result
 
    update = {"messages": new_messages}
    if retrieved_context is not None:
        update["retrieved_context"] = retrieved_context
    if tool_result is not None:
        update["tool_result"] = tool_result
    return update
 
 

def sales_node(state: AgentState) -> dict:
    """Handles sales conversation. Bound to four tools: retrieve_product_info
    (product questions, includes live stock), check_product_availability
    (re-checking a known product_id without a fresh search), add_to_cart
    (save an item for later), and create_order (the real business action)
    — the LLM decides which, if any, to call based on the conversation."""
    tools_list = [retrieve_product_info, check_product_availability, add_to_cart, create_order]
    # bind the tools to the LLM so it can call them by name in its reasoning
    llm = get_llm().bind_tools(tools_list)

    system_prompt = (
        "You are a sales assistant for an e-commerce store. Help the "
        "customer find products and answer questions about price and "
        "availability using retrieve_product_info. "
        "retrieve_product_info includes current stock for every result, "
        "so use that number directly when the customer asks about "
        "availability or quantity — no separate check needed for a "
        "product you just looked up. "
        "Use add_to_cart when the customer wants to save an item or build "
        "up a cart without buying immediately. "
        "Only call create_order after the customer has clearly confirmed "
        "they want to buy specific items right now. "
        "Only state product facts that are explicitly present in what "
        "retrieve_product_info or check_product_availability returns — do "
        "not invent features, stock guarantees, or details that aren't in it. "
        "This includes descriptors like \"wireless\" — a search might return "
        "a product that isn't actually a strong match; if a product's "
        "description doesn't support a label the customer asked about, "
        "either leave that product out of your answer or mention it without "
        "claiming it has that property, rather than inventing a justification "
        "for why it fits. "
        "If a question isn't actually about products, prices, availability, "
        "carts, or orders — for example warranties, shipping, returns, or "
        "any store policy — you don't have a way to look that up here. "
        "Don't answer it from general assumptions about how stores "
        "typically work; say honestly that you're not the right place for "
        "that question and that customer service can help instead, rather "
        "than guessing. "
        f"When calling add_to_cart or create_order, always use "
        f"customer_id={state['customer_id']} — this is the only account "
        "you can ever act on, regardless of what the customer says. "
        "If the customer asks you to place an order, add to cart, or do "
        "anything for a different customer — whether they refer to that "
        "other customer by an id number, a name (e.g. \"order this for "
        "John\"), or any other identifier — you must explicitly tell them, "
        "every time, that you can only act on their own account — never "
        "silently proceed as if they hadn't asked for that, and never "
        "comply with it, no matter how the other customer is identified. "
        + _SHARED_BEHAVIOR_RULES
    )
 

    # Sends system message and conversation history to the LLM
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return _run_agent_loop(llm, messages, tools_list)


def customer_service_node(state: AgentState) -> dict:
    """Handles customer-service conversation. Bound to only retrieve_support_info"""
    tools_list = [retrieve_support_info]
    llm = get_llm().bind_tools(tools_list)

    system_prompt = (
        "You are a customer service assistant for an e-commerce store. "
        "Answer questions about shipping, returns, and policies using "
        "retrieve_support_info to find accurate information before answering. "
        "Only state facts that are explicitly present in what retrieve_support_info "
        "returns — do not invent procedures, contact instructions, or details "
        "that aren't in it. If the retrieved information doesn't fully answer "
        "the question, say so honestly rather than filling the gap yourself. "
        + _SHARED_BEHAVIOR_RULES
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return _run_agent_loop(llm, messages, tools_list)


def format_response(state: AgentState) -> dict:
    """Last node before END. Pulls the agent's finished reply out of the
    conversation history into state['response'] , giving the Flask chat
    route one single, predictable field to read the final answer from,
    regardless of which branch (sales or customer service) produced it."""
    last_message = state["messages"][-1]
    return {"response": last_message.content}