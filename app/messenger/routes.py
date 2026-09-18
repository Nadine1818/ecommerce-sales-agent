"""Webhook for the Messenger integration. Meta calls this route
directly (not the frontend) , GET for the one-time verification
handshake, POST for every actual message event."""

import os
import re
 
import requests
from flask import request
from langchain_core.messages import AIMessage, HumanMessage
 
from app.agent import compiled_graph
from app.extensions import db
from app.messenger import messenger_bp
from app.models import User

# Meta sends this token in the GET request to verify that we control this server.
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
# Meta sends this token in the POST request to authenticate us as the Page sending the reply.
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN")

# Every reply sent goes through this one Graph API endpoint, regardless
# of which customer it's going to , the recipient is specified per-request
# in the JSON body, not the URL.
GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"

# Matches the pre-filled text from our m.me?text=LINK-<user_id> deep link
# Anchored + case-insensitive so a customer
# editing the pre-filled text slightly (extra spaces, lowercase) still
# links correctly, without accidentally matching unrelated messages.
LINK_CODE_RE = re.compile(r"^\s*LINK-(\d+)\s*$", re.IGNORECASE)

# Conversation history is stored in memory here, keyed by customer_id.
CONVERSATION_HISTORY: dict[int, list[dict]] = {}

# Meta calls this once, when registering the webhook URL in the dashboard,
# It sends three query params; if the token matches theirs, echo back their challenge value verbatim.
@messenger_bp.route("/webhook", methods=["GET"])
def verify():
    # mode: identifies the type of request , challenge: random number meta generates 
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200

    return "Verification failed", 403

# Meta calls this every time a customer sends the Page a message.
# Meta sends post request to ../messenger/webhook with a JSON body containing the message event(s).
@messenger_bp.route("/webhook", methods=["POST"])
def receive():
    data = request.get_json()
 
    # Loop rather than assume [0]: a single webhook call can batch
    # multiple entries, and each entry can batch multiple messaging
    # events, even though in practice one test message is almost always
    # exactly one of each.
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            handle_messaging_event(event)
 
    return "ok", 200

def link_account(sender_id: str, user_id: int) -> "User | None":
    candidate = User.query.get(user_id)
    if candidate:
        candidate.messenger_psid = sender_id
        db.session.commit()
        send_message(sender_id, f"Hi, {candidate.name}! How can I help?")
    return candidate

def handle_messaging_event(event: dict):
    sender_id = event.get("sender", {}).get("id")
    if not sender_id:
        return
 
    message = event.get("message", {})
    text = message.get("text")
 
    # If the sender_id is already linked to a user, we can proceed to handle the message.
    user = User.query.filter_by(messenger_psid=sender_id).first()
 
    if not user:
        # Path 1 (primary, invisible to the customer): Meta's own ?ref=
        # referral, reliable for a genuinely first-time conversation with
        # the Page.
        referral = event.get("referral")
        ref_value = referral.get("ref") if referral else None
        if ref_value and ref_value.isdigit():
            user = link_account(sender_id, int(ref_value))
            if user:
                return
 
        # Path 2 (silent fallback): a typed "LINK-<user_id>" message,
        # for accounts where referral delivery didn't come through (see
        # app/chat/routes.py). Not shown or suggested to most customers
        # — only relevant if Path 1 fails.
        if text:
            match = LINK_CODE_RE.match(text)
            if match:
                user = link_account(sender_id, int(match.group(1)))
                if not user:
                    send_message(
                        sender_id,
                        "That link doesn't seem to match an account. "
                        "Please open 'Chat on Messenger' fresh from your "
                        "account page while logged in.",
                    )
                # Either outcome: this message WAS the link code, not a
                # real question — stop here.
                return
 
    # Still not linked — neither path above matched. We have no
    # customer_id to run the agent with, so we can't proceed.
    if not user:
        send_message(
            sender_id,
            "To chat with me here, please open the 'Chat on Messenger' "
            "link from your account page on our site while logged in — "
            "that lets me look up your orders and cart.",
        )
        return
 
    # Non-text events (stickers, quick replies, attachments) don't have a
    # "text" key — skip them for now rather than crash.
    if not text:
        return
 
    history = CONVERSATION_HISTORY.get(user.id, [])
 
    messages = []
    for turn in history:
        if turn["role"] == "human":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=text))
 
    state = {
        "messages": messages,
        "customer_id": user.id,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
 
    result = compiled_graph.invoke(state)
 
    history.append({"role": "human", "content": text})
    history.append({"role": "ai", "content": result["response"]})
    CONVERSATION_HISTORY[user.id] = history
 
    send_message(sender_id, result["response"])
 
# send post request the graph api url 
def send_message(recipient_id: str, text: str):
    """Sends a text message to one Messenger user via the Graph API,
    authenticated as our Page using PAGE_ACCESS_TOKEN."""
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
    }
    params = {"access_token": PAGE_ACCESS_TOKEN}
 
    response = requests.post(GRAPH_API_URL, params=params, json=payload)
 
    # Not raising on failure here , a failed send shouldn't crash the
    # webhook handler and cause Meta to retry the whole event. Printing
    # for now so failures are visible while we're testing manually.
    if response.status_code != 200:
        print("Failed to send message:", response.status_code, response.text)