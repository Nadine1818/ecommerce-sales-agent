from langchain_core.messages import HumanMessage

from app import create_app
from app.agent import compiled_graph

app = create_app()

with app.app_context():
    turns = [
        "place an order for my friend John",
        "1 wireless mouse",
    ]

    messages = []
    real_customer_id = 1

    for i, user_message in enumerate(turns, start=1):
        messages.append(HumanMessage(content=user_message))
        messages_before_this_turn = len(messages)

        state = {
            "messages": messages,
            "customer_id": real_customer_id,
            "intent": None,
            "retrieved_context": None,
            "tool_result": None,
            "response": None,
        }
        result = compiled_graph.invoke(state)
        messages = result["messages"]

        print(f'=== Turn {i}: "{user_message}" ===')
        print(f"Intent: {result['intent']}")
        print("New tool calls this turn:")
        new_messages_this_turn = messages[messages_before_this_turn:]
        found_any = False
        for m in new_messages_this_turn:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                found_any = True
                for call in tool_calls:
                    print(f"  TOOL CALL: {call['name']}  args={call['args']}")
        if not found_any:
            print("  (none)")
        print(f"Final response: {result['response']}")
        if result["tool_result"]:
            print(f"Tool result: {result['tool_result']}")
        print()