"""Unit tests for app/agent/tools.py's real business-action tools:
create_order, add_to_cart, and check_product_availability. (The retrieve_*
tools are thin wrappers over retrieve(), which needs the embedding
model/vector store and is exercised by tests/manual/check_retrieve.py
instead.)"""

from app.agent.tools import add_to_cart, check_product_availability, create_order
from app.extensions import db
from app.models import Cart, CartItem, Order, Product


def test_create_order_success(sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    result = create_order.invoke(
        {"customer_id": customer.id, "items": [{"product_id": product.id, "quantity": 2}]}
    )

    assert "error" not in result
    assert result["total_price"] == 50.0
    assert result["status"] == "confirmed"

    order = db.session.get(Order, result["order_id"])
    assert order.customer_id == customer.id
    assert len(order.items) == 1
    assert order.items[0].quantity == 2
    # price_at_order is a snapshot, taken independently of product.price.
    assert order.items[0].price_at_order == 25.0


def test_create_order_decrements_stock(sample_data):
    product = sample_data["product"]

    create_order.invoke(
        {"customer_id": sample_data["customer"].id, "items": [{"product_id": product.id, "quantity": 2}]}
    )

    refreshed = db.session.get(Product, product.id)
    assert refreshed.stock_quantity == 3


def test_create_order_multiple_items(sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    second = Product(name="USB Keyboard", price=40.0, stock_quantity=3, category=sample_data["category"])
    db.session.add(second)
    db.session.commit()

    result = create_order.invoke(
        {
            "customer_id": customer.id,
            "items": [
                {"product_id": product.id, "quantity": 1},
                {"product_id": second.id, "quantity": 2},
            ],
        }
    )

    assert result["total_price"] == 25.0 + 2 * 40.0
    order = db.session.get(Order, result["order_id"])
    assert len(order.items) == 2


def test_create_order_nonexistent_product(sample_data):
    result = create_order.invoke(
        {"customer_id": sample_data["customer"].id, "items": [{"product_id": 9999, "quantity": 1}]}
    )

    assert "error" in result
    assert "does not exist" in result["error"]
    assert Order.query.count() == 0


def test_create_order_insufficient_stock(sample_data):
    product = sample_data["product"]

    result = create_order.invoke(
        {"customer_id": sample_data["customer"].id, "items": [{"product_id": product.id, "quantity": 999}]}
    )

    assert "error" in result
    assert "Not enough stock" in result["error"]
    # stock must be untouched, and no order row left behind.
    assert db.session.get(Product, product.id).stock_quantity == 5
    assert Order.query.count() == 0


def test_create_order_partial_failure_does_not_commit_earlier_items(sample_data):
    """items are validated and applied in a single loop before the one
    commit at the end — if a later item fails, nothing from that same
    call should have been persisted, including the stock decrement for
    the item that was fine on its own."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    bad_product = Product(name="Out of Stock Item", price=10.0, stock_quantity=0, category=sample_data["category"])
    db.session.add(bad_product)
    db.session.commit()

    result = create_order.invoke(
        {
            "customer_id": customer.id,
            "items": [
                {"product_id": product.id, "quantity": 1},
                {"product_id": bad_product.id, "quantity": 1},
            ],
        }
    )

    assert "error" in result
    assert Order.query.count() == 0
    assert db.session.get(Product, product.id).stock_quantity == 5


def test_add_to_cart_creates_cart_and_item(sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    result = add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 2})

    assert "error" not in result
    assert result["quantity_in_cart"] == 2

    cart = Cart.query.filter_by(user_id=customer.id).first()
    assert cart is not None
    assert len(cart.items) == 1
    assert cart.items[0].quantity == 2


def test_add_to_cart_does_not_touch_stock(sample_data):
    """Unlike create_order, adding to a cart is a soft action — it must
    not decrement real inventory, since an abandoned cart shouldn't lock
    stock away from other customers."""
    customer = sample_data["customer"]
    product = sample_data["product"]

    add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 2})

    refreshed = db.session.get(Product, product.id)
    assert refreshed.stock_quantity == 5


def test_add_to_cart_same_product_twice_increments_not_duplicates(sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 1})
    result = add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 2})

    assert result["quantity_in_cart"] == 3
    cart = Cart.query.filter_by(user_id=customer.id).first()
    assert len(cart.items) == 1  # one row, not two


def test_add_to_cart_nonexistent_product(sample_data):
    result = add_to_cart.invoke({"customer_id": sample_data["customer"].id, "product_id": 9999, "quantity": 1})

    assert "error" in result
    assert "does not exist" in result["error"]
    assert Cart.query.count() == 0


def test_add_to_cart_insufficient_stock(sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]

    result = add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 999})

    assert "error" in result
    assert "Not enough stock" in result["error"]
    # nothing should have been created for a rejected add
    assert Cart.query.count() == 0


def test_add_to_cart_reuses_existing_cart_across_calls(sample_data):
    """The second call shouldn't create a second Cart row for the same
    customer — Cart.user_id is unique, so this also guards against a
    silent integrity-constraint failure being the reason it works."""
    customer = sample_data["customer"]
    product = sample_data["product"]

    add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 1})
    add_to_cart.invoke({"customer_id": customer.id, "product_id": product.id, "quantity": 1})

    assert Cart.query.filter_by(user_id=customer.id).count() == 1


def test_check_product_availability_existing_product(sample_data):
    product = sample_data["product"]

    result = check_product_availability.invoke({"product_id": product.id})

    assert result["product_id"] == product.id
    assert result["name"] == product.name
    assert result["stock_quantity"] == 5
    assert result["in_stock"] is True


def test_check_product_availability_zero_stock_reports_out_of_stock(sample_data):
    product = sample_data["product"]
    product.stock_quantity = 0
    db.session.commit()

    result = check_product_availability.invoke({"product_id": product.id})

    assert result["stock_quantity"] == 0
    assert result["in_stock"] is False


def test_check_product_availability_nonexistent_product(sample_data):
    result = check_product_availability.invoke({"product_id": 9999})

    assert "error" in result
    assert "does not exist" in result["error"]