"""Routes for the customer-facing chat demo. Conversation history is kept
in the Flask session (one browser session = one ongoing conversation),
storing only the customer-facing human/ai turns as plain dicts, not the
internal tool-calling messages, which are regenerated fresh each turn and
don't need to persist across requests."""

# request gives access to the incoming http request
from flask import jsonify, render_template, request, session
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import compiled_graph
from app.chat import chat_bp
from app.models import User

# the route "/" serves the chat UI
@chat_bp.route("/")
def index():
    # fetches all customers from the database to populate the customer picker in the UI
    customers = User.query.filter_by(role="customer").all()
    return render_template("chat.html", customers=customers)

# the route "/send" handles incoming messages from the chat UI, 
# runs them through the agent graph, and returns the agent's response
@chat_bp.route("/send", methods=["POST"])
def send():
    data = request.get_json()
    customer_id = data.get("customer_id")
    message = (data.get("message") or "").strip()

    if not customer_id or not message:
        # 400 bad request if either customer_id or message is missing
        return jsonify({"error": "customer_id and message are required"}), 400

    customer_id = int(customer_id)

    # Switching to a different customer (or starting fresh) resets the
    # stored conversation history, as a conversation only makes sense in the context of
    # one customer at a time.
    if session.get("customer_id") != customer_id:
        session["customer_id"] = customer_id
        session["history"] = []

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

    result = compiled_graph.invoke(state)

    # Persist only the new human/ai pair. result["messages"] also contains
    # the tool-calling scaffolding (the AIMessage that requested a tool,
    # the ToolMessage with its result) but that's not saved here
    history.append({"role": "human", "content": message})
    history.append({"role": "ai", "content": result["response"]})
    session["history"] = history

    # intent is included in case i want to display it in the UI for debugging
    return jsonify({"response": result["response"], "intent": result["intent"]})