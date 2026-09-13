"""Unit tests for app/agent/tools.py's create_order — the only tool that
performs a real business action (the retrieve_* tools are thin wrappers
over retrieve(), which needs the embedding model/vector store and is
exercised by tests/manual/check_retrieve.py instead)."""

from app.agent.tools import create_order
from app.extensions import db
from app.models import Order, Product


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
