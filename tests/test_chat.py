"""Tests for the customer-facing chat routes (app/chat/routes.py):
history persistence in the session, guest vs logged-in behavior, the
requires_login signal, and error handling when the agent graph itself
fails. The LangGraph agent is mocked at the compiled_graph.invoke
boundary — the graph's own internal behavior is covered by
tests/test_agent.py."""

import pytest

from app.chat import routes as chat_routes


class FakeCompiledGraph:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.invocations = []

    def invoke(self, state):
        self.invocations.append(state)
        if self._exc is not None:
            raise self._exc
        return self._result


def _patch_graph(monkeypatch, **kwargs):
    fake = FakeCompiledGraph(**kwargs)
    monkeypatch.setattr(chat_routes, "compiled_graph", fake)
    return fake


@pytest.fixture()
def client(app):
    return app.test_client()


def _default_result(response="Hello!", intent="sales", tool_result=None):
    return {
        "messages": [],
        "response": response,
        "intent": intent,
        "tool_result": tool_result,
        "retrieved_context": None,
    }


# -------------------------------------------------------------------- index

def test_index_renders_for_guest(client):
    response = client.get("/chat/")
    assert response.status_code == 200
    assert b"Browsing as guest" in response.data


def test_index_shows_empty_history_initially(client):
    response = client.get("/chat/")
    assert b"Ask about our products" in response.data


def test_index_replays_stored_history(client, monkeypatch):
    _patch_graph(monkeypatch, result=_default_result(response="Hi there!"))
    client.post("/chat/send", json={"message": "hello"})

    response = client.get("/chat/")

    assert b"hello" in response.data
    assert b"Hi there!" in response.data


def test_index_includes_messenger_link_when_configured_and_logged_in(client, app, monkeypatch):
    monkeypatch.setenv("MESSENGER_PAGE_USERNAME", "QuickShelf")
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["name"] = "Bob"
        sess["role"] = "customer"

    response = client.get("/chat/")

    assert b"m.me/QuickShelf?ref=1" in response.data


def test_index_no_messenger_link_for_guest_even_if_configured(client, monkeypatch):
    monkeypatch.setenv("MESSENGER_PAGE_USERNAME", "QuickShelf")

    response = client.get("/chat/")

    assert b"m.me/QuickShelf" not in response.data


def test_index_no_messenger_link_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("MESSENGER_PAGE_USERNAME", raising=False)
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["name"] = "Bob"
        sess["role"] = "customer"

    response = client.get("/chat/")

    assert b"m.me/" not in response.data


# --------------------------------------------------------------------- send

def test_send_requires_message_field(client):
    response = client.post("/chat/send", json={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "message is required"


def test_send_rejects_blank_message(client):
    response = client.post("/chat/send", json={"message": "   "})
    assert response.status_code == 400


def test_send_happy_path_returns_response_and_intent(client, monkeypatch):
    _patch_graph(monkeypatch, result=_default_result(response="We have mice in stock.", intent="sales"))

    response = client.post("/chat/send", json={"message": "any mice?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["response"] == "We have mice in stock."
    assert body["intent"] == "sales"
    assert body["requires_login"] is False


def test_send_persists_human_and_ai_turns_in_session(client, monkeypatch):
    _patch_graph(monkeypatch, result=_default_result(response="Sure thing."))

    client.post("/chat/send", json={"message": "hi"})

    with client.session_transaction() as sess:
        history = sess["history"]
        assert history == [
            {"role": "human", "content": "hi"},
            {"role": "ai", "content": "Sure thing."},
        ]


def test_send_second_message_appends_to_existing_history(client, monkeypatch):
    _patch_graph(monkeypatch, result=_default_result(response="First reply."))
    client.post("/chat/send", json={"message": "first"})

    _patch_graph(monkeypatch, result=_default_result(response="Second reply."))
    client.post("/chat/send", json={"message": "second"})

    with client.session_transaction() as sess:
        assert len(sess["history"]) == 4
        assert sess["history"][2] == {"role": "human", "content": "second"}
        assert sess["history"][3] == {"role": "ai", "content": "Second reply."}


def test_send_passes_none_customer_id_for_guest(client, monkeypatch):
    fake = _patch_graph(monkeypatch, result=_default_result())
    client.post("/chat/send", json={"message": "hi"})

    assert fake.invocations[0]["customer_id"] is None


def test_send_passes_logged_in_customer_id(client, monkeypatch):
    fake = _patch_graph(monkeypatch, result=_default_result())
    with client.session_transaction() as sess:
        sess["user_id"] = 7
        sess["name"] = "Bob"

    client.post("/chat/send", json={"message": "hi"})

    assert fake.invocations[0]["customer_id"] == 7


def test_send_rebuilds_message_history_for_graph_state(client, monkeypatch):
    """Prior turns stored in the session must be replayed into the graph
    state as real LangChain message objects, in order, before the new
    message."""
    from langchain_core.messages import AIMessage, HumanMessage

    _patch_graph(monkeypatch, result=_default_result(response="Reply 1"))
    client.post("/chat/send", json={"message": "first question"})

    fake2 = _patch_graph(monkeypatch, result=_default_result(response="Reply 2"))
    client.post("/chat/send", json={"message": "second question"})

    sent = fake2.invocations[0]["messages"]
    assert isinstance(sent[0], HumanMessage) and sent[0].content == "first question"
    assert isinstance(sent[1], AIMessage) and sent[1].content == "Reply 1"
    assert isinstance(sent[2], HumanMessage) and sent[2].content == "second question"


def test_send_requires_login_flag_surfaces_when_tool_result_says_so(client, monkeypatch):
    _patch_graph(
        monkeypatch,
        result=_default_result(response="Please log in to continue.", tool_result={"requires_login": True}),
    )

    response = client.post("/chat/send", json={"message": "I want to order it"})

    assert response.get_json()["requires_login"] is True


def test_send_requires_login_flag_false_when_tool_result_is_none(client, monkeypatch):
    _patch_graph(monkeypatch, result=_default_result(tool_result=None))

    response = client.post("/chat/send", json={"message": "hi"})

    assert response.get_json()["requires_login"] is False


def test_send_agent_exception_returns_502_and_does_not_save_history(client, monkeypatch):
    _patch_graph(monkeypatch, exc=RuntimeError("LLM is down"))

    response = client.post("/chat/send", json={"message": "hi"})

    assert response.status_code == 502
    assert "temporarily unavailable" in response.get_json()["error"]
    with client.session_transaction() as sess:
        assert "history" not in sess


# ------------------------------------------------------- admin can't shop

def test_index_redirects_admin_to_dashboard(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["name"] = "Root"
        sess["role"] = "admin"

    response = client.get("/chat/")

    assert response.status_code == 302
    assert response.headers["Location"] == "/dashboard/"


def test_send_rejects_admin_session(client, monkeypatch):
    fake = _patch_graph(monkeypatch, result=_default_result())
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["name"] = "Root"
        sess["role"] = "admin"

    response = client.post("/chat/send", json={"message": "hi"})

    assert response.status_code == 403
    assert fake.invocations == []  # never reached the agent at all


# ------------------------------------------------------------- root redirect

def test_root_redirects_guest_to_chat(client):
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"


def test_root_redirects_admin_to_dashboard_via_chat(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["name"] = "Root"
        sess["role"] = "admin"

    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert response.request.path == "/dashboard/products"