"""Routes for the customer-facing chat demo. Conversation history is kept
in the Flask session (one browser session = one ongoing conversation),
storing only the customer-facing human/ai turns as plain dicts, not the
internal tool-calling messages, which are regenerated fresh each turn and
don't need to persist across requests."""

# request gives access to the incoming http request
from flask import current_app, jsonify, redirect, render_template, request, session, url_for
from langchain_core.messages import AIMessage, HumanMessage
 
from app.agent import compiled_graph
from app.auth.decorators import login_required
from app.chat import chat_bp
import os

# the route "/" serves the chat UI
@chat_bp.route("/")
def index():
    # This UI (and cart/orders, linked from its nav) is for guests and
    # customers only. An admin landing here — e.g. by navigating to
    # /chat/ directly after logging in, rather than being sent here by
    # the login flow — should see their own dashboard instead, not the
    # shopping interface: /cart/ and /orders/ already refuse an admin
    # via login_required(role="customer"), but this route had no
    # equivalent check at all until now.
    if session.get("role") == "admin":
        return redirect(url_for("dashboard.index"))

    page_username = os.environ.get("MESSENGER_PAGE_USERNAME")
    messenger_link = None
    if page_username and "user_id" in session:
        messenger_link = f"https://m.me/{page_username}?ref={session['user_id']}"
 
    return render_template(
        "chat.html",
        user_name=session.get("name"),
        history=session.get("history", []),
        messenger_link=messenger_link,
    )


# the route "/send" handles incoming messages from the chat UI, 
# runs them through the agent graph, and returns the agent's response
@chat_bp.route("/send", methods=["POST"])
def send():
    # Defense in depth alongside index()'s redirect: index() stops an
    # admin from ever loading this page in the first place, but this
    # guards the API endpoint itself too — e.g. a tab left open from
    # before switching sessions — so an admin's own account can never
    # be used as customer_id to run the sales flow (and place a real
    # order) through the chat backend, even if the page somehow got
    # loaded.
    if session.get("role") == "admin":
        return jsonify({"error": "Admins can't use the customer chat. Use the dashboard instead."}), 403

    data = request.get_json()
    message = (data.get("message") or "").strip()


    if not message:
        return jsonify({"error": "message is required"}), 400
 
    # customer_id comes from the logged-in session
    customer_id = session.get("user_id")  # None for guest users, or the user's id for logged-in customers

    history = session.get("history", [])

    # gets the conversation history from the session, or an empty list if none exists
    history = session.get("history", [])

    # Rebuild LangChain message objects from the plain dicts stored in
    # the session. Only human/ai turns are kept
    messages = []
    for turn in history:
        if turn["role"] == "human":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=message))

    state = {
        "messages": messages,
        "customer_id": customer_id,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }

    try:
        result = compiled_graph.invoke(state)
    except Exception:
        current_app.logger.exception("Agent invocation failed")
        return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 502

    # Persist only the new human/ai pair. result["messages"] also contains
    # the tool-calling scaffolding (the AIMessage that requested a tool,
    # the ToolMessage with its result) but that's not saved here
    history.append({"role": "human", "content": message})
    history.append({"role": "ai", "content": result["response"]})
    session["history"] = history

    # tool_result carries {"requires_login": True} when the guest-mode
    # request_login tool fired, so the frontend knows to show a login
    # prompt instead of just displaying the agent's text reply
    requires_login = bool((result.get("tool_result") or {}).get("requires_login"))

    # intent is included in case i want to display it in the UI for debugging
    return jsonify({"response": result["response"], "intent": result["intent"], "requires_login": requires_login})