"""Tests for the read-only order history page (app/orders/routes.py),
including edge cases: cross-customer isolation, ordering, a price
snapshot that must not follow later product price changes, a deleted
product referenced by an old order, and an unrecognized status value."""

import pytest

from app.extensions import db
from app.models import Order, OrderItem, Product, User


@pytest.fixture()
def client(app):
    return app.test_client()


def _login(client, user):
    with client.session_transaction() as sess:
        sess["user_id"] = user.id
        sess["role"] = user.role
        sess["name"] = user.name


def _other_customer(email="other@example.com"):
    other = User(name="Other", email=email, role="customer")
    other.set_password("pw")
    db.session.add(other)
    db.session.commit()
    return other


def _make_order(customer_id, product, quantity=1, price_at_order=None, status="confirmed"):
    order = Order(customer_id=customer_id, status=status, total_price=0.0)
    price = product.price if price_at_order is None else price_at_order
    order.items.append(OrderItem(product=product, quantity=quantity, price_at_order=price))
    order.total_price = price * quantity
    db.session.add(order)
    db.session.commit()
    return order


# --------------------------------------------------------------- auth gate

def test_orders_index_requires_login(client):
    response = client.get("/orders/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_admin_role_is_locked_out_of_orders(client, app_context):
    admin = User(name="Root", email="admin-orders@example.com", role="admin")
    admin.set_password("pw")
    db.session.add(admin)
    db.session.commit()
    _login(client, admin)

    response = client.get("/orders/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


# --------------------------------------------------------------------- GET

def test_orders_index_empty_state(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert response.status_code == 200
    assert b"haven" in response.data.lower()


def test_orders_index_shows_order_with_items_and_total(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    _make_order(customer.id, product, quantity=2, price_at_order=25.0)

    _login(client, customer)
    response = client.get("/orders/")

    assert response.status_code == 200
    assert b"Wireless Mouse" in response.data
    assert b"Order #" in response.data
    assert response.data.count(b"50.00") >= 1  # line total and order total


def test_orders_index_only_shows_current_users_orders(client, app_context, sample_data):
    other = _other_customer("other-orders@example.com")
    _make_order(other.id, sample_data["product"])

    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert response.status_code == 200
    assert b"haven" in response.data.lower()  # sees the empty state, not the other user's order


def test_orders_index_newest_first(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    older = _make_order(customer.id, product, quantity=1)
    newer = _make_order(customer.id, product, quantity=1)

    _login(client, customer)
    response = client.get("/orders/")

    body = response.data
    pos_newer = body.index(f"Order #{newer.id}".encode())
    pos_older = body.index(f"Order #{older.id}".encode())
    assert pos_newer < pos_older


def test_orders_index_price_snapshot_ignores_later_price_change(client, app_context, sample_data):
    """price_at_order is a snapshot; the page must reflect what the
    customer actually paid, not the product's current live price."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    _make_order(customer.id, product, quantity=1, price_at_order=25.0)

    product.price = 999.0  # price changes after the order was placed
    db.session.commit()

    _login(client, customer)
    response = client.get("/orders/")

    assert b"25.00" in response.data
    assert b"999.00" not in response.data


def test_orders_index_survives_deleted_product_reference(client, app_context, sample_data):
    """An order item keeps its own price/quantity snapshot even if the
    underlying product is later deleted; the page must render a fallback
    label instead of crashing on item.product being None."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    order = _make_order(customer.id, product, quantity=3, price_at_order=25.0)
    product_id = product.id

    db.session.delete(product)
    db.session.commit()

    _login(client, customer)
    response = client.get("/orders/")

    assert response.status_code == 200
    assert f"Product #{product_id}".encode() in response.data
    assert b"75.00" in response.data  # 3 * 25.00 snapshot price still shown


def test_orders_index_multiple_items_per_order(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    second = Product(name="USB Keyboard", price=40.0, stock_quantity=3, category=sample_data["category"])
    db.session.add(second)
    db.session.flush()

    order = Order(customer_id=customer.id, status="confirmed")
    order.items.append(OrderItem(product=product, quantity=1, price_at_order=25.0))
    order.items.append(OrderItem(product=second, quantity=2, price_at_order=40.0))
    order.total_price = 25.0 + 2 * 40.0
    db.session.add(order)
    db.session.commit()

    _login(client, customer)
    response = client.get("/orders/")

    assert b"Wireless Mouse" in response.data
    assert b"USB Keyboard" in response.data
    assert b"105.00" in response.data


@pytest.mark.parametrize("status", ["pending", "confirmed", "cancelled", "shipped", "Weird Status!"])
def test_orders_index_renders_for_any_status_value(client, app_context, sample_data, status):
    """Status drives a CSS class (status-<value>); an unrecognized or
    oddly-shaped value must not break rendering, just fall back to
    unstyled text."""
    _make_order(sample_data["customer"].id, sample_data["product"], status=status)

    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert response.status_code == 200
    assert status.capitalize().encode() in response.data


def test_orders_index_xss_shaped_status_is_escaped(client, app_context, sample_data):
    payload = "<script>alert(1)</script>"
    _make_order(sample_data["customer"].id, sample_data["product"], status=payload)

    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert response.status_code == 200
    assert payload.encode() not in response.data


def test_orders_index_order_with_no_items(client, app_context, sample_data):
    """Shouldn't normally happen via create_order, but the page must not
    crash on an order with an empty items list."""
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=0.0)
    db.session.add(order)
    db.session.commit()

    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert response.status_code == 200
    assert b"Order #" in response.data


def test_orders_index_zero_total_formats_as_two_decimals(client, app_context, sample_data):
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=0.0)
    db.session.add(order)
    db.session.commit()

    _login(client, sample_data["customer"])
    response = client.get("/orders/")

    assert b"$0.00" in response.data
