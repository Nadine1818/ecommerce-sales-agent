# Tools the agent's LLM can call. Two are thin wrappers around retrieve()
# (scoped to different content types), and one is the real business
# action the agent can take."""

from langchain_core.tools import tool
 
from app.rag import retrieve
 

# hardcode the item types for the two retrieve tools, so the agent doesn't have to guess which one to call for a given question
@tool
def retrieve_product_info(query: str) -> str:
    """Search the product catalog for items matching the customer's question.
    Use this for anything about what products are available, their prices,
    descriptions, or categories. Each result includes the product's id —
    use that exact id when calling create_order."""
    matches = retrieve(query, top_k=3, item_type="product")
    if not matches:
        return "No matching products found."
    lines = []
    for m in matches:
        product_id = m["metadata"].get("id", "unknown")
        lines.append(f"- [product_id: {product_id}] {m['text']}")
    return "\n".join(lines)

@tool
def retrieve_support_info(query: str) -> str:
    """Search FAQs and store policies (shipping, returns, delivery) for
    anything related to the customer's question. Use this for
    customer-service questions, not product questions."""
    matches = retrieve(query, top_k=3, item_type=["faq", "policy"])
    if not matches:
        return "No matching support information found."
    return "\n".join(f"- {m['text']}" for m in matches)


@tool
def create_order(customer_id: int, items: list[dict]) -> dict:
    """Places a real order for the customer. Only call this after the
    customer has clearly confirmed they want to buy specific items —
    never call it just because products were discussed.
 
    items: a list of dicts, each shaped like {"product_id": <int>, "quantity": <int>}.
 
    Returns a dict describing the result: either the created order's id,
    total price, and status, or an error message if a product doesn't
    exist or isn't in stock.
    """
    from app.extensions import db
    from app.models import Order, OrderItem, Product

    order = Order(customer_id=customer_id, status="confirmed")
    total_price = 0.0

    for item in items:
        product = db.session.get(Product, item["product_id"])
        if product is None:
            return {"error": f"Product {item['product_id']} does not exist."}
        if product.stock_quantity < item["quantity"]:
            return {
                "error": f"Not enough stock for {product.name} "
                f"(requested {item['quantity']}, have {product.stock_quantity})."
            }

        order_item = OrderItem(
            product=product,
            quantity=item["quantity"],
            price_at_order=product.price,
        )
        order.items.append(order_item)
        product.stock_quantity -= item["quantity"]
        total_price += product.price * item["quantity"]

    order.total_price = total_price

    db.session.add(order)
    db.session.commit()

    return {
        "order_id": order.id,
        "total_price": order.total_price,
        "status": order.status,
    }