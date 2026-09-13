from langchain_core.messages import HumanMessage

from app import create_app
from app.agent.graph import compiled_graph
from app.models import Order, Product

app = create_app()

with app.app_context():
    product_before = Product.query.filter_by(name="Wireless Mouse").first()
    print(f"Stock before: {product_before.stock_quantity}")

    # Turn 1 — ask about the product, no order yet.
    state = {
        "messages": [HumanMessage(content="do you have a wireless mouse?")],
        "customer_id": 1,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
    result = compiled_graph.invoke(state)
    print("\n=== Turn 1 ===")
    print("Response:", result["response"])

    # Turn 2 — confirm the purchase. Carrying forward result["messages"]
    # (the full conversation so far, including the tool call/result from
    # turn 1) is what lets the LLM understand "yes, order it" refers to
    # the wireless mouse just discussed.
    state2 = {
        "messages": result["messages"] + [HumanMessage(content="yes, please order 1 wireless mouse for me")],
        "customer_id": 1,
        "intent": None,
        "retrieved_context": None,
        "tool_result": None,
        "response": None,
    }
    result2 = compiled_graph.invoke(state2)
    print("\n=== Turn 2 ===")
    print("Response:", result2["response"])
    print("Tool result:", result2["tool_result"])

    # Verify against the real database — not just trusting the agent's
    # claimed response, but checking the actual rows.
    product_after = Product.query.filter_by(name="Wireless Mouse").first()
    print(f"\nStock after: {product_after.stock_quantity}")

    if result2["tool_result"] and "order_id" in result2["tool_result"]:
        from app.extensions import db
        order = db.session.get(Order, result2["tool_result"]["order_id"])
        print(f"Order {order.id}: status={order.status}, total=${order.total_price}, items={len(order.items)}")