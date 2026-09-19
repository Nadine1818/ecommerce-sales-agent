"""Tests for the LangGraph agent: node logic (app/agent/nodes.py) and the
compiled graph's routing (app/agent/graph.py).

The LLM itself is never called for real — every test swaps
app.agent.nodes.get_llm for a small FakeLLM that plays back a scripted
sequence of responses. This keeps the suite fast, deterministic, and
independent of a GROQ_API_KEY, while still exercising the *real* graph
wiring, the *real* tool-calling loop (_run_agent_loop/_execute_tool_calls),
and the *real* business-action tools (add_to_cart/create_order/etc.)
against a genuine (in-memory, per-test) database — only the model's text
output is faked, not the surrounding system.

retrieve_product_info / retrieve_support_info are intentionally avoided
in the scripted tool calls here (they hit the real RAG/embedding
pipeline, already covered by tests/test_rag.py) so this file stays fast.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent import nodes as nodes_module
from app.agent.graph import compiled_graph
from app.agent.nodes import (
    _execute_tool_calls,
    _run_agent_loop,
    classify_intent,
    customer_service_node,
    format_response,
    route_by_intent,
    sales_node,
)
from app.models import Cart, Order


class FakeLLM:
    """Stands in for ChatGroq. `responses` is a queue: each call to
    .invoke() pops the next entry, returning it directly if it's an
    AIMessage, or raising it if it's an Exception (so retry behavior can
    be exercised too)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.invocations = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_llm(monkeypatch, fake_llm):
    monkeypatch.setattr(nodes_module, "get_llm", lambda: fake_llm)


def _no_sleep(monkeypatch):
    """classify_intent/_run_agent_loop retry with time.sleep() between
    attempts — patched to a no-op so retry tests don't actually wait."""
    monkeypatch.setattr(nodes_module.time, "sleep", lambda _seconds: None)


# ------------------------------------------------------------ classify_intent

def test_classify_intent_sales(monkeypatch, app_context):
    _patch_llm(monkeypatch, FakeLLM([AIMessage(content="sales")]))

    result = classify_intent({"messages": [HumanMessage(content="do you have mice?")]})

    assert result == {"intent": "sales"}


def test_classify_intent_customer_service(monkeypatch, app_context):
    _patch_llm(monkeypatch, FakeLLM([AIMessage(content="customer_service")]))

    result = classify_intent({"messages": [HumanMessage(content="what's your return policy?")]})

    assert result == {"intent": "customer_service"}


def test_classify_intent_strips_and_lowercases_response(monkeypatch, app_context):
    _patch_llm(monkeypatch, FakeLLM([AIMessage(content="  SALES  ")]))

    result = classify_intent({"messages": [HumanMessage(content="price of mouse?")]})

    assert result == {"intent": "sales"}


def test_classify_intent_unrecognized_word_defaults_to_customer_service(monkeypatch, app_context):
    _patch_llm(monkeypatch, FakeLLM([AIMessage(content="banana")]))

    result = classify_intent({"messages": [HumanMessage(content="anything")]})

    assert result == {"intent": "customer_service"}


def test_classify_intent_llm_failure_after_retries_defaults_to_customer_service(monkeypatch, app_context):
    _no_sleep(monkeypatch)
    _patch_llm(monkeypatch, FakeLLM([RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]))

    result = classify_intent({"messages": [HumanMessage(content="anything")]})

    assert result == {"intent": "customer_service"}


def test_classify_intent_recovers_after_transient_failure(monkeypatch, app_context):
    """The retry wrapper should let a single flaky call through without
    falling back at all."""
    _no_sleep(monkeypatch)
    _patch_llm(monkeypatch, FakeLLM([RuntimeError("transient"), AIMessage(content="sales")]))

    result = classify_intent({"messages": [HumanMessage(content="anything")]})

    assert result == {"intent": "sales"}


def test_classify_intent_sends_full_conversation_history(monkeypatch, app_context):
    """A bare '1' answering a prior quantity question is meaningless
    without context — the whole history must reach the LLM, not just the
    latest message."""
    fake = FakeLLM([AIMessage(content="sales")])
    _patch_llm(monkeypatch, fake)

    history = [
        HumanMessage(content="do you have mice?"),
        AIMessage(content="Yes, how many would you like?"),
        HumanMessage(content="1"),
    ]
    classify_intent({"messages": history})

    sent_messages = fake.invocations[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[1:] == history


# ---------------------------------------------------------------- route_by_intent

@pytest.mark.parametrize("intent", ["sales", "customer_service"])
def test_route_by_intent_returns_state_intent(intent):
    assert route_by_intent({"intent": intent}) == intent


# ---------------------------------------------------------------- format_response

def test_format_response_extracts_last_message_content():
    state = {"messages": [HumanMessage(content="hi"), AIMessage(content="Hello there!")]}
    assert format_response(state) == {"response": "Hello there!"}


# ---------------------------------------------------------------- tool binding

def test_sales_node_guest_gets_request_login_not_cart_or_order_tools(monkeypatch, app_context):
    fake = FakeLLM([AIMessage(content="I can help you browse our products.")])
    _patch_llm(monkeypatch, fake)

    sales_node({"messages": [HumanMessage(content="hi")], "customer_id": None})

    tool_names = {t.name for t in fake.bound_tools}
    assert tool_names == {"retrieve_product_info", "check_product_availability", "request_login"}
    assert "add_to_cart" not in tool_names
    assert "create_order" not in tool_names


def test_sales_node_logged_in_customer_gets_cart_and_order_tools(monkeypatch, app_context):
    fake = FakeLLM([AIMessage(content="Sure, I can help with that.")])
    _patch_llm(monkeypatch, fake)

    sales_node({"messages": [HumanMessage(content="hi")], "customer_id": 42})

    tool_names = {t.name for t in fake.bound_tools}
    assert tool_names == {"retrieve_product_info", "check_product_availability", "add_to_cart", "create_order"}
    assert "request_login" not in tool_names


def test_customer_service_node_only_gets_support_tool(monkeypatch, app_context):
    fake = FakeLLM([AIMessage(content="Our return window is 30 days.")])
    _patch_llm(monkeypatch, fake)

    customer_service_node({"messages": [HumanMessage(content="what's your return policy?")], "customer_id": None})

    tool_names = {t.name for t in fake.bound_tools}
    assert tool_names == {"retrieve_support_info"}


# -------------------------------------------------------- guest tool-call gating

def test_sales_node_guest_attempting_add_to_cart_gets_tool_error_not_a_real_add(monkeypatch, app_context):
    """add_to_cart isn't bound for a guest, but nothing stops the LLM
    from hallucinating a call to it anyway — _execute_tool_calls must
    reject it with an error message the LLM can read, not crash, and
    definitely not actually add anything to a cart."""
    hallucinated_call = AIMessage(
        content="",
        tool_calls=[{"name": "add_to_cart", "args": {"customer_id": 1, "product_id": 1, "quantity": 1}, "id": "call_1"}],
    )
    final = AIMessage(content="You'll need to log in first.")
    fake = FakeLLM([hallucinated_call, final])
    _patch_llm(monkeypatch, fake)

    result = sales_node({"messages": [HumanMessage(content="add a mouse to my cart")], "customer_id": None})

    # messages[0] is the AI's tool-call request itself; the ToolMessage
    # with the resulting error comes right after it.
    tool_message = result["messages"][1]
    assert "no tool named 'add_to_cart'" in tool_message.content
    assert Cart.query.count() == 0


# ------------------------------------------------------------- real tool calls

def test_sales_node_executes_add_to_cart_for_real(monkeypatch, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "add_to_cart",
            "args": {"customer_id": customer.id, "product_id": product.id, "quantity": 2},
            "id": "call_1",
        }],
    )
    final = AIMessage(content="Added 2 to your cart.")
    fake = FakeLLM([tool_call, final])
    _patch_llm(monkeypatch, fake)

    result = sales_node({"messages": [HumanMessage(content="add 2 mice to my cart")], "customer_id": customer.id})

    cart = Cart.query.filter_by(user_id=customer.id).first()
    assert cart is not None
    assert cart.items[0].quantity == 2
    # the AI's final reply (no more tool calls) ends up as the last message
    assert result["messages"][-1].content == "Added 2 to your cart."


def test_sales_node_executes_create_order_for_real(monkeypatch, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    tool_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "create_order",
            "args": {"customer_id": customer.id, "items": [{"product_id": product.id, "quantity": 1}]},
            "id": "call_1",
        }],
    )
    final = AIMessage(content="Your order is placed!")
    fake = FakeLLM([tool_call, final])
    _patch_llm(monkeypatch, fake)

    result = sales_node({"messages": [HumanMessage(content="order one mouse")], "customer_id": customer.id})

    assert Order.query.count() == 1
    assert result["tool_result"]["order_id"] == Order.query.first().id


def test_sales_node_guest_confirms_purchase_triggers_request_login(monkeypatch, app_context):
    tool_call = AIMessage(content="", tool_calls=[{"name": "request_login", "args": {}, "id": "call_1"}])
    final = AIMessage(content="Please log in to continue — I'll pick this back up once you're in.")
    fake = FakeLLM([tool_call, final])
    _patch_llm(monkeypatch, fake)

    result = sales_node({"messages": [HumanMessage(content="I want to order it")], "customer_id": None})

    assert result["tool_result"] == {"requires_login": True}


# ------------------------------------------------------------- agent loop cap

def test_run_agent_loop_stops_when_llm_stops_requesting_tools(monkeypatch, app_context, sample_data):
    fake = FakeLLM([AIMessage(content="Here's what I found.")])
    update = _run_agent_loop(fake, [SystemMessage(content="sys")], tools_list=[])

    assert len(fake.invocations) == 1
    assert update["messages"][-1].content == "Here's what I found."


def test_run_agent_loop_respects_max_iterations_safety_cap(monkeypatch, app_context, sample_data):
    """An LLM that never stops requesting tools must not loop forever —
    max_iterations caps it, and the loop must not crash even though the
    final iteration still has unresolved tool_calls."""
    from app.agent.tools import check_product_availability

    product = sample_data["product"]
    endless_call = AIMessage(
        content="",
        tool_calls=[{"name": "check_product_availability", "args": {"product_id": product.id}, "id": "call_x"}],
    )
    fake = FakeLLM([endless_call, endless_call, endless_call, endless_call, endless_call])

    update = _run_agent_loop(fake, [SystemMessage(content="sys")], tools_list=[check_product_availability], max_iterations=3)

    assert len(fake.invocations) == 3  # capped, not 5
    assert len(fake.responses) == 2  # two scripted responses never consumed
    # 3 iterations, each appending one AI tool-call request + its ToolMessage result
    assert len(update["messages"]) == 6


def test_execute_tool_calls_multiple_calls_in_one_ai_message(app_context, sample_data):
    """The LLM can request more than one tool call in a single turn —
    both must run and both must produce a ToolMessage."""
    from app.agent.tools import check_product_availability

    product = sample_data["product"]
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "check_product_availability", "args": {"product_id": product.id}, "id": "call_1"},
            {"name": "check_product_availability", "args": {"product_id": 9999}, "id": "call_2"},
        ],
    )

    tool_messages, _, _ = _execute_tool_calls(ai_message, [check_product_availability])

    assert len(tool_messages) == 2
    assert tool_messages[0].tool_call_id == "call_1"
    assert tool_messages[1].tool_call_id == "call_2"
    assert "does not exist" in tool_messages[1].content


# ----------------------------------------------------------------- full graph

def test_compiled_graph_sales_path_end_to_end(monkeypatch, app_context, sample_data):
    classify = FakeLLM([AIMessage(content="sales")])
    sales_reply = FakeLLM([AIMessage(content="We have a wireless mouse for $25.")])

    # classify_intent and sales_node both call get_llm(); the first call
    # (classification) must get `classify`, every call after must get
    # `sales_reply` — a simple counter distinguishes the two.
    call_count = {"n": 0}

    def get_llm_sequence():
        call_count["n"] += 1
        return classify if call_count["n"] == 1 else sales_reply

    monkeypatch.setattr(nodes_module, "get_llm", get_llm_sequence)

    state = {
        "messages": [HumanMessage(content="do you have a wireless mouse?")],
        "customer_id": sample_data["customer"].id,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
    result = compiled_graph.invoke(state)

    assert result["intent"] == "sales"
    assert result["response"] == "We have a wireless mouse for $25."


def test_compiled_graph_customer_service_path_end_to_end(monkeypatch, app_context):
    classify = FakeLLM([AIMessage(content="customer_service")])
    cs_reply = FakeLLM([AIMessage(content="Returns are accepted within 30 days.")])

    call_count = {"n": 0}

    def get_llm_sequence():
        call_count["n"] += 1
        return classify if call_count["n"] == 1 else cs_reply

    monkeypatch.setattr(nodes_module, "get_llm", get_llm_sequence)

    state = {
        "messages": [HumanMessage(content="what's your return policy?")],
        "customer_id": None,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
    result = compiled_graph.invoke(state)

    assert result["intent"] == "customer_service"
    assert result["response"] == "Returns are accepted within 30 days."


def test_compiled_graph_sales_order_flow_updates_database(monkeypatch, app_context, sample_data):
    """Full graph, real DB side effect: classify -> sales -> create_order
    -> format_response, verified against the actual Order row."""
    customer = sample_data["customer"]
    product = sample_data["product"]

    classify = FakeLLM([AIMessage(content="sales")])
    order_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "create_order",
            "args": {"customer_id": customer.id, "items": [{"product_id": product.id, "quantity": 1}]},
            "id": "call_1",
        }],
    )
    final = AIMessage(content="Order placed!")
    sales_reply = FakeLLM([order_call, final])

    call_count = {"n": 0}

    def get_llm_sequence():
        call_count["n"] += 1
        return classify if call_count["n"] == 1 else sales_reply

    monkeypatch.setattr(nodes_module, "get_llm", get_llm_sequence)

    state = {
        "messages": [HumanMessage(content="order 1 wireless mouse")],
        "customer_id": customer.id,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
    result = compiled_graph.invoke(state)

    assert result["response"] == "Order placed!"
    assert result["tool_result"]["order_id"] == Order.query.first().id
    assert Order.query.count() == 1
