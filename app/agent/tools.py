# Tools the agent's LLM can call. Two are thin wrappers around retrieve()
# (scoped to different content types), check_product_availability,
# add_to_cart, and create_order are the real business actions """

from langchain_core.tools import tool
 
from app.rag import retrieve
 

# hardcode the item types for the two retrieve tools, so the agent doesn't have to guess which one to call for a given question
@tool
def retrieve_product_info(query: str) -> str:
    """Search the product catalog for items matching the customer's question.
    Use this for anything about what products are available, their prices,
    descriptions, or categories — this already includes current stock, so
    you don't need a separate check for a product you just retrieved. Each
    result includes the product's id — use that exact id when calling
    add_to_cart or create_order."""
    matches = retrieve(query, top_k=3, item_type="product")
    if not matches:
        return "No matching products found."

    from app.extensions import db
    from app.models import Product
    # add stock quantity to the output for each product, so the agent can answer questions about availability 
    # without needing to call check_product_availability for products it just retrieved.
    lines = []
    for m in matches:
        product_id = m["metadata"].get("id", "unknown")
        stock_note = ""
        if product_id != "unknown":
            product = db.session.get(Product, int(product_id))
            if product is not None:
                stock_note = f" | In stock: {product.stock_quantity}"
        lines.append(f"- [product_id: {product_id}] {m['text']}{stock_note}")
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

# used when the customer asks about a product's stock or availability,
# but looks it up by id in case the customer already knows the product id from a previous retrieve_product_info result.
@tool
def check_product_availability(product_id: int) -> dict:
    """Re-check current stock for a product you already know the id of
    from earlier in this conversation, without doing a fresh product
    search. retrieve_product_info already includes stock for anything
    you look up, so only use this if the customer asks again about a
    product mentioned earlier and you don't want to re-search for it."""
    product = db_session_get_product(product_id)
    if product is None:
        return {"error": f"Product {product_id} does not exist."}
 
    return {
        "product_id": product.id,
        "name": product.name,
        "stock_quantity": product.stock_quantity,
        "in_stock": product.stock_quantity > 0,
    }

def db_session_get_product(product_id: int):
    # Small shared helper so check_product_availability and add_to_cart
    # both fetch a product the same, non-deprecated way.
    from app.extensions import db
    from app.models import Product
 
    return db.session.get(Product, product_id)

# atomic guarded decrement is only handled in create_order, not add_to_cart,
# since add_to_cart is just a "save for later" action and doesn't actually commit to buying the product yet.
@tool
def add_to_cart(customer_id: int, product_id: int, quantity: int) -> dict:
    """Adds a product to the customer's persistent cart — separate from
    placing an order via create_order. Use this when the customer wants
    to save an item for later or build up a cart, rather than buy it
    right now. If the product is already in their cart, increases the
    quantity instead of adding a duplicate line.
 
    Returns the cart item's resulting quantity, or an error message if
    the product doesn't exist or there isn't enough stock.
    """
    from app.extensions import db
    from app.models import Cart, CartItem
 
    product = db_session_get_product(product_id)
    if product is None:
        return {"error": f"Product {product_id} does not exist."}
    if product.stock_quantity < quantity:
        return {
            "error": f"Not enough stock for {product.name} "
            f"(requested {quantity}, have {product.stock_quantity})."
        }
 
    # Lazily create the customer's cart the first time they add
    # something , no precreation step needed
    cart = Cart.query.filter_by(user_id=customer_id).first()
    if cart is None:
        cart = Cart(user_id=customer_id)
        db.session.add(cart)
        try:
            db.session.flush()
        except IntegrityError:
            db.session.rollback()
            cart = Cart.query.filter_by(user_id=customer_id).first()
            
    existing_item = CartItem.query.filter_by(cart_id=cart.id, product_id=product_id).first()
    if existing_item:
        existing_item.quantity += quantity
    else:
        existing_item = CartItem(cart=cart, product=product, quantity=quantity)
        db.session.add(existing_item)
 
    db.session.commit()
 
    return {
        "product_id": product.id,
        "product_name": product.name,
        "quantity_in_cart": existing_item.quantity,
    }
 

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
    from sqlalchemy import update
    from app.extensions import db
    from app.models import Order, OrderItem, Product

    order = Order(customer_id=customer_id, status="confirmed")
    total_price = 0.0

    for item in items:
        product = db_session_get_product(item["product_id"])
        if product is None:
            db.session.rollback()
            return {"error": f"Product {item['product_id']} does not exist."}
        # atomic guarded decrement of stock_quantity to avoid race conditions if multiple orders are placed at the same time by different customers. 
        result = db.session.execute(
            update(Product)
            .where(Product.id == item["product_id"], Product.stock_quantity >= item["quantity"])
            .values(stock_quantity=Product.stock_quantity - item["quantity"])
        )
        # if no rows were updated, it means there wasn't enough stock to fulfill the order. Roll back the transaction and return an error message.
        if result.rowcount == 0:
            db.session.rollback()
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
        total_price += product.price * item["quantity"]
 
    order.total_price = total_price

    db.session.add(order)
    db.session.commit()

    return {
        "order_id": order.id,
        "total_price": order.total_price,
        "status": order.status,
    }