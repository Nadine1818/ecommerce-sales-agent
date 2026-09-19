"""Tests for the Facebook Messenger bonus integration
(app/messenger/routes.py): the webhook verification handshake, account
linking (both the referral path and the typed LINK-<id> fallback),
message handling for linked/unlinked users, and the outbound
send_message() call to the Graph API.

compiled_graph.invoke and requests.post are both mocked — this is a
webhook handler test, not a test of the agent itself (see
tests/test_agent.py) or of real network calls to Meta.

VERIFY_TOKEN/PAGE_ACCESS_TOKEN are read from the environment at *module
import time* in app/messenger/routes.py (plain `os.environ.get(...)`
module-level constants), so monkeypatch.setenv after the fact has no
effect on them — tests instead patch the module attributes directly.
"""

import pytest

from app.messenger import routes as messenger_routes
from app.models import User
from app.extensions import db


class FakeCompiledGraph:
    def __init__(self, response="Hello!"):
        self.response = response
        self.invocations = []

    def invoke(self, state):
        self.invocations.append(state)
        return {"messages": [], "response": self.response, "intent": "sales", "tool_result": None, "retrieved_context": None}


class FakeResponse:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def _clean_conversation_history():
    """CONVERSATION_HISTORY is a plain module-level dict, shared across
    every request in the process (not per Flask session) — it must be
    reset around each test or state leaks between tests."""
    messenger_routes.CONVERSATION_HISTORY.clear()
    yield
    messenger_routes.CONVERSATION_HISTORY.clear()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def sent_messages(monkeypatch):
    """Captures every outbound call to the Graph API without making a
    real network request."""
    calls = []

    def fake_post(url, params=None, json=None):
        calls.append({"url": url, "params": params, "json": json})
        return FakeResponse(status_code=200)

    monkeypatch.setattr(messenger_routes.requests, "post", fake_post)
    return calls


@pytest.fixture()
def fake_graph(monkeypatch):
    fake = FakeCompiledGraph()
    monkeypatch.setattr(messenger_routes, "compiled_graph", fake)
    return fake


# -------------------------------------------------------------- verify (GET)

def test_verify_correct_token_echoes_challenge(client, monkeypatch):
    monkeypatch.setattr(messenger_routes, "VERIFY_TOKEN", "correct-token")

    response = client.get(
        "/messenger/webhook",
        query_string={"hub.mode": "subscribe", "hub.verify_token": "correct-token", "hub.challenge": "12345"},
    )

    assert response.status_code == 200
    assert response.data.decode() == "12345"


def test_verify_wrong_token_rejected(client, monkeypatch):
    monkeypatch.setattr(messenger_routes, "VERIFY_TOKEN", "correct-token")

    response = client.get(
        "/messenger/webhook",
        query_string={"hub.mode": "subscribe", "hub.verify_token": "wrong-token", "hub.challenge": "12345"},
    )

    assert response.status_code == 403


def test_verify_wrong_mode_rejected(client, monkeypatch):
    monkeypatch.setattr(messenger_routes, "VERIFY_TOKEN", "correct-token")

    response = client.get(
        "/messenger/webhook",
        query_string={"hub.mode": "unsubscribe", "hub.verify_token": "correct-token", "hub.challenge": "12345"},
    )

    assert response.status_code == 403


def test_verify_missing_params_rejected(client, monkeypatch):
    monkeypatch.setattr(messenger_routes, "VERIFY_TOKEN", "correct-token")

    response = client.get("/messenger/webhook")

    assert response.status_code == 403


# -------------------------------------------------------------- link_account

def test_link_account_existing_user_sets_psid_and_greets(app_context, sample_data, sent_messages):
    customer = sample_data["customer"]

    result = messenger_routes.link_account("psid-123", customer.id)

    assert result is not None
    assert result.id == customer.id
    refreshed = db.session.get(User, customer.id)
    assert refreshed.messenger_psid == "psid-123"
    assert len(sent_messages) == 1
    assert "Test Customer" in sent_messages[0]["json"]["message"]["text"]


def test_link_account_nonexistent_user_returns_none_and_sends_nothing(app_context, sent_messages):
    result = messenger_routes.link_account("psid-999", 9999)

    assert result is None
    assert sent_messages == []


# --------------------------------------------------------- handle_messaging_event

def test_event_with_no_sender_id_is_ignored(app_context, fake_graph, sent_messages):
    messenger_routes.handle_messaging_event({"message": {"text": "hi"}})

    assert fake_graph.invocations == []
    assert sent_messages == []


def test_already_linked_user_goes_straight_to_agent(app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]
    customer.messenger_psid = "psid-linked"
    db.session.commit()

    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-linked"}, "message": {"text": "do you have mice?"}}
    )

    assert len(fake_graph.invocations) == 1
    assert fake_graph.invocations[0]["customer_id"] == customer.id
    assert len(sent_messages) == 1
    assert sent_messages[0]["json"]["message"]["text"] == "Hello!"


def test_linked_user_conversation_history_persists_across_events(app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]
    customer.messenger_psid = "psid-linked"
    db.session.commit()

    messenger_routes.handle_messaging_event({"sender": {"id": "psid-linked"}, "message": {"text": "first"}})
    messenger_routes.handle_messaging_event({"sender": {"id": "psid-linked"}, "message": {"text": "second"}})

    history = messenger_routes.CONVERSATION_HISTORY[customer.id]
    assert [h["content"] for h in history] == ["first", "Hello!", "second", "Hello!"]
    # second call's graph state included the first turn's messages
    second_call_messages = fake_graph.invocations[1]["messages"]
    assert len(second_call_messages) == 3  # human, ai, human


def test_referral_links_new_user_and_does_not_call_agent_this_turn(app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]
    assert customer.messenger_psid is None

    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-new"}, "referral": {"ref": str(customer.id)}, "message": {"text": "hi"}}
    )

    refreshed = db.session.get(User, customer.id)
    assert refreshed.messenger_psid == "psid-new"
    assert fake_graph.invocations == []  # the referral message itself isn't treated as a real question
    assert len(sent_messages) == 1
    assert "Test Customer" in sent_messages[0]["json"]["message"]["text"]


def test_link_code_fallback_links_user(app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]

    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-new"}, "message": {"text": f"LINK-{customer.id}"}}
    )

    refreshed = db.session.get(User, customer.id)
    assert refreshed.messenger_psid == "psid-new"
    assert fake_graph.invocations == []


def test_link_code_fallback_case_and_whitespace_insensitive(app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]

    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-new"}, "message": {"text": f"  link-{customer.id}  "}}
    )

    refreshed = db.session.get(User, customer.id)
    assert refreshed.messenger_psid == "psid-new"


def test_link_code_fallback_unknown_user_id_sends_error_and_stops(app_context, fake_graph, sent_messages):
    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-new"}, "message": {"text": "LINK-99999"}}
    )

    assert fake_graph.invocations == []
    assert len(sent_messages) == 1
    assert "doesn't seem to match" in sent_messages[0]["json"]["message"]["text"]


def test_unlinked_user_no_referral_no_link_code_prompts_to_link(app_context, fake_graph, sent_messages):
    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-unknown"}, "message": {"text": "hello, is anyone there?"}}
    )

    assert fake_graph.invocations == []
    assert len(sent_messages) == 1
    assert "Chat on Messenger" in sent_messages[0]["json"]["message"]["text"]


def test_linked_user_non_text_event_is_skipped(app_context, sample_data, fake_graph, sent_messages):
    """Stickers/attachments have no "text" key — must not crash, and
    must not reach the agent (nothing meaningful to send it)."""
    customer = sample_data["customer"]
    customer.messenger_psid = "psid-linked"
    db.session.commit()

    messenger_routes.handle_messaging_event({"sender": {"id": "psid-linked"}, "message": {"sticker_id": 123}})

    assert fake_graph.invocations == []
    assert sent_messages == []


def test_referral_ref_non_numeric_falls_through_to_unlinked_prompt(app_context, fake_graph, sent_messages):
    messenger_routes.handle_messaging_event(
        {"sender": {"id": "psid-new"}, "referral": {"ref": "not-a-number"}, "message": {"text": "hi"}}
    )

    assert fake_graph.invocations == []
    assert len(sent_messages) == 1
    assert "Chat on Messenger" in sent_messages[0]["json"]["message"]["text"]


# ------------------------------------------------------------------ webhook POST

def test_receive_dispatches_every_event_in_every_entry(client, app_context, sample_data, fake_graph, sent_messages):
    customer = sample_data["customer"]
    customer.messenger_psid = "psid-a"
    other = User(name="Other", email="other-messenger@example.com", role="customer", messenger_psid="psid-b")
    other.set_password("pw")
    db.session.add(other)
    db.session.commit()

    payload = {
        "entry": [
            {"messaging": [{"sender": {"id": "psid-a"}, "message": {"text": "hi from a"}}]},
            {"messaging": [{"sender": {"id": "psid-b"}, "message": {"text": "hi from b"}}]},
        ]
    }
    response = client.post("/messenger/webhook", json=payload)

    assert response.status_code == 200
    assert response.data == b"ok"
    assert len(fake_graph.invocations) == 2


def test_receive_empty_payload_does_not_crash(client, fake_graph, sent_messages):
    response = client.post("/messenger/webhook", json={})

    assert response.status_code == 200
    assert fake_graph.invocations == []


def test_receive_agent_failure_degrades_gracefully(client, app_context, sample_data, monkeypatch, sent_messages):
    """An agent failure (e.g. Groq being down or rate-limited) must not
    propagate as an unhandled exception — mirrors app/chat/routes.py's
    /send, which catches the same failure and degrades gracefully
    instead of letting Meta receive a 500 and redeliver the whole event.
    The customer gets a friendly fallback reply, and the webhook still
    reports success to Meta so it doesn't retry."""
    customer = sample_data["customer"]
    customer.messenger_psid = "psid-fail"
    db.session.commit()

    class BoomGraph:
        def invoke(self, state):
            raise RuntimeError("LLM down")

    monkeypatch.setattr(messenger_routes, "compiled_graph", BoomGraph())

    response = client.post(
        "/messenger/webhook",
        json={"entry": [{"messaging": [{"sender": {"id": "psid-fail"}, "message": {"text": "hi"}}]}]},
    )

    assert response.status_code == 200
    assert response.data == b"ok"
    assert len(sent_messages) == 1
    assert "trouble right now" in sent_messages[0]["json"]["message"]["text"]
    # the failed turn isn't recorded — nothing to persist for a question
    # the agent never actually answered
    assert customer.id not in messenger_routes.CONVERSATION_HISTORY


# ------------------------------------------------------------------ send_message

def test_send_message_posts_expected_payload(app_context, monkeypatch, sent_messages):
    monkeypatch.setattr(messenger_routes, "PAGE_ACCESS_TOKEN", "test-page-token")

    messenger_routes.send_message("recipient-1", "Hello there")

    assert len(sent_messages) == 1
    call = sent_messages[0]
    assert call["url"] == messenger_routes.GRAPH_API_URL
    assert call["params"] == {"access_token": "test-page-token"}
    assert call["json"] == {"recipient": {"id": "recipient-1"}, "message": {"text": "Hello there"}}


def test_send_message_failure_does_not_raise(app_context, monkeypatch, capsys):
    monkeypatch.setattr(
        messenger_routes.requests, "post", lambda *a, **k: FakeResponse(status_code=500, text="server error")
    )

    messenger_routes.send_message("recipient-1", "Hello there")  # must not raise

    assert "Failed to send message" in capsys.readouterr().out
