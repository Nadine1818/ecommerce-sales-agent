"""Webhook for the Messenger integration. Meta calls this route
directly (not the frontend) , GET for the one-time verification
handshake, POST for every actual message event."""

import os

from flask import request
import requests
import requests

from app.messenger import messenger_bp

# Meta sends this token in the GET request to verify that we control this server.
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
# Meta sends this token in the POST request to authenticate us as the Page sending the reply.
PAGE_ACCESS_TOKEN = os.environ.get("PAGE_ACCESS_TOKEN")

# Every reply sent goes through this one Graph API endpoint, regardless
# of which customer it's going to , the recipient is specified per-request
# in the JSON body, not the URL.
GRAPH_API_URL = "https://graph.facebook.com/v21.0/me/messages"

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
            sender_id = event.get("sender", {}).get("id")
 
            # Non-text events (stickers, quick replies, attachments) don't
            # have a "text" key , skipped them for now rather than crash.
            message = event.get("message", {})
            text = message.get("text")
 
            if sender_id and text:
                send_message(sender_id, f"You said: {text}")
 
    return "ok", 200
 
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