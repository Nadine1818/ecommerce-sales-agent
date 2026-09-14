"""Tests for the shopping cart page and its GET/update/remove/checkout
routes (app/cart/routes.py), including edge cases: cross-customer item
access, invalid/out-of-stock quantities, an empty or nonexistent cart,
a cart item whose product was deleted out from under it, and checkout
failing partway through a multi-item cart."""

import pytest

from app.extensions import db
from app.models import Cart, CartItem, Order, Product, User


@pytest.fixture()
def client(app):
    return app.test_client()


def _login(client, user):
    with client.session_transaction() as sess:
        sess["user_id"] = user.id
        sess["role"] = user.role
        sess["name"] = user.name


@pytest.fixture()
def cart_with_item(app_context, sample_data):
    """sample_data's customer with a cart containing 2x the sample
    product (which has 5 units in stock)."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product=product, quantity=2)
    db.session.add(item)
    db.session.commit()
    return {**sample_data, "cart": cart, "item": item}


def _other_customer(email="other@example.com"):
    other = User(name="Other", email=email, role="customer")
    other.set_password("pw")
    db.session.add(other)
    db.session.commit()
    return other


# --------------------------------------------------------------- auth gate

def test_cart_index_requires_login(client):
    response = client.get("/cart/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_cart_update_requires_login(client):
    response = client.post("/cart/update/1", data={"quantity": "1"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_cart_remove_requires_login(client):
    response = client.post("/cart/remove/1")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_cart_checkout_requires_login(client):
    response = client.post("/cart/checkout")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_admin_role_is_locked_out_of_cart(client, app_context):
    admin = User(name="Root", email="admin-cart@example.com", role="admin")
    admin.set_password("pw")
    db.session.add(admin)
    db.session.commit()
    _login(client, admin)

    response = client.get("/cart/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


# --------------------------------------------------------------------- GET

def test_cart_index_empty_when_no_cart_row_exists(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.get("/cart/")
    assert response.status_code == 200
    assert b"Your cart is empty" in response.data


def test_cart_index_empty_when_cart_exists_but_has_no_items(client, app_context, sample_data):
    db.session.add(Cart(user_id=sample_data["customer"].id))
    db.session.commit()
    _login(client, sample_data["customer"])

    response = client.get("/cart/")
    assert response.status_code == 200
    assert b"Your cart is empty" in response.data


def test_cart_index_shows_items_and_computes_total(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    response = client.get("/cart/")

    assert response.status_code == 200
    assert b"Wireless Mouse" in response.data
    # 2 * 25.0 = 50.00, both as the line subtotal and the grand total
    assert response.data.count(b"50.00") >= 2


def test_cart_index_only_shows_current_users_cart(client, app_context, sample_data):
    other = _other_customer()
    other_cart = Cart(user_id=other.id)
    db.session.add(other_cart)
    db.session.flush()
    db.session.add(CartItem(cart=other_cart, product=sample_data["product"], quantity=1))
    db.session.commit()

    _login(client, sample_data["customer"])
    response = client.get("/cart/")

    assert response.status_code == 200
    assert b"Your cart is empty" in response.data


def test_cart_index_product_name_is_escaped(client, app_context, sample_data):
    xss_product = Product(
        name="<script>alert(1)</script>", price=5.0, stock_quantity=10, category=sample_data["category"]
    )
    db.session.add(xss_product)
    db.session.flush()
    cart = Cart(user_id=sample_data["customer"].id)
    db.session.add(cart)
    db.session.flush()
    db.session.add(CartItem(cart=cart, product=xss_product, quantity=1))
    db.session.commit()

    _login(client, sample_data["customer"])
    response = client.get("/cart/")

    assert b"<script>alert(1)</script>" not in response.data
    assert b"&lt;script&gt;" in response.data


def test_cart_index_survives_item_whose_product_was_deleted(client, app_context, sample_data):
    """SQLite doesn't enforce the cart_items -> products foreign key by
    default, so a product can be deleted while still referenced by a
    cart item. The page must render (skipping that item's price) rather
    than crash with an AttributeError on None.price."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product_id=product.id, quantity=1)
    db.session.add(item)
    db.session.commit()

    db.session.delete(product)
    db.session.commit()

    _login(client, customer)
    response = client.get("/cart/")

    assert response.status_code == 200
    assert b"no longer available" in response.data
    assert b"$0.00" in response.data  # total excludes the orphaned item


def test_cart_index_mixes_valid_and_orphaned_items_in_total(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    doomed = Product(name="Doomed Widget", price=15.0, stock_quantity=2, category=sample_data["category"])
    db.session.add(doomed)
    db.session.flush()

    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    db.session.add(CartItem(cart=cart, product=product, quantity=1))  # 25.0
    orphan = CartItem(cart=cart, product_id=doomed.id, quantity=1)
    db.session.add(orphan)
    db.session.commit()

    db.session.delete(doomed)
    db.session.commit()

    _login(client, customer)
    response = client.get("/cart/")

    assert response.status_code == 200
    assert b"Wireless Mouse" in response.data
    assert b"no longer available" in response.data
    # total counts only the surviving product
    assert b"$25.00" in response.data


# ------------------------------------------------------------------ update

def test_update_quantity_happy_path(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/update/{item_id}", data={"quantity": "4"})

    assert response.status_code == 302
    assert response.headers["Location"] == "/cart/"
    assert db.session.get(CartItem, item_id).quantity == 4


@pytest.mark.parametrize("quantity", ["0", "-1", "abc", ""])
def test_update_quantity_rejects_invalid_values(client, cart_with_item, quantity):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/update/{item_id}", data={"quantity": quantity}, follow_redirects=True)

    assert response.status_code == 200
    assert b"Quantity must be at least 1." in response.data
    assert db.session.get(CartItem, item_id).quantity == 2  # unchanged


def test_update_quantity_rejects_missing_field(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/update/{item_id}", data={}, follow_redirects=True)

    assert b"Quantity must be at least 1." in response.data
    assert db.session.get(CartItem, item_id).quantity == 2


def test_update_quantity_rejects_more_than_stock(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id  # product has 5 in stock

    response = client.post(f"/cart/update/{item_id}", data={"quantity": "999"}, follow_redirects=True)

    assert b"left in stock" in response.data
    assert db.session.get(CartItem, item_id).quantity == 2


def test_update_quantity_allows_exact_stock_boundary(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id  # product has 5 in stock

    response = client.post(f"/cart/update/{item_id}", data={"quantity": "5"})

    assert response.status_code == 302
    assert db.session.get(CartItem, item_id).quantity == 5


def test_update_nonexistent_item_returns_404(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.post("/cart/update/99999", data={"quantity": "1"})
    assert response.status_code == 404


def test_update_someone_elses_item_returns_404_and_does_not_modify(client, app_context, cart_with_item):
    other = _other_customer("other-update@example.com")
    _login(client, other)
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/update/{item_id}", data={"quantity": "99"})

    assert response.status_code == 404
    assert db.session.get(CartItem, item_id).quantity == 2


def test_update_item_whose_product_was_deleted_does_not_crash(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product_id=product.id, quantity=1)
    db.session.add(item)
    db.session.commit()
    item_id = item.id

    db.session.delete(product)
    db.session.commit()

    _login(client, customer)
    response = client.post(f"/cart/update/{item_id}", data={"quantity": "3"}, follow_redirects=True)

    assert response.status_code == 200
    assert b"no longer available" in response.data
    assert db.session.get(CartItem, item_id).quantity == 1  # untouched


# ------------------------------------------------------------------ remove

def test_remove_happy_path(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/remove/{item_id}", follow_redirects=True)

    assert response.status_code == 200
    assert b"Removed Wireless Mouse from your cart." in response.data
    assert db.session.get(CartItem, item_id) is None


def test_remove_nonexistent_item_returns_404(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.post("/cart/remove/99999")
    assert response.status_code == 404


def test_remove_someone_elses_item_returns_404_and_does_not_delete(client, app_context, cart_with_item):
    other = _other_customer("other-remove@example.com")
    _login(client, other)
    item_id = cart_with_item["item"].id

    response = client.post(f"/cart/remove/{item_id}")

    assert response.status_code == 404
    assert db.session.get(CartItem, item_id) is not None


def test_remove_item_whose_product_was_deleted_does_not_crash(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product_id=product.id, quantity=1)
    db.session.add(item)
    db.session.commit()
    item_id = item.id

    db.session.delete(product)
    db.session.commit()

    _login(client, customer)
    response = client.post(f"/cart/remove/{item_id}", follow_redirects=True)

    assert response.status_code == 200
    assert b"Removed item from your cart." in response.data
    assert db.session.get(CartItem, item_id) is None


# --------------------------------------------------------------- checkout

def test_checkout_empty_cart_no_cart_row(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.post("/cart/checkout", follow_redirects=True)

    assert response.status_code == 200
    assert b"Your cart is empty." in response.data
    assert Order.query.count() == 0


def test_checkout_cart_row_exists_but_has_no_items(client, app_context, sample_data):
    db.session.add(Cart(user_id=sample_data["customer"].id))
    db.session.commit()
    _login(client, sample_data["customer"])

    response = client.post("/cart/checkout", follow_redirects=True)

    assert b"Your cart is empty." in response.data
    assert Order.query.count() == 0


def test_checkout_happy_path_creates_order_and_clears_cart(client, cart_with_item):
    _login(client, cart_with_item["customer"])

    response = client.post("/cart/checkout", follow_redirects=True)

    assert response.status_code == 200
    assert b"order-card" in response.data  # landed on the orders page
    assert Order.query.count() == 1
    order = Order.query.first()
    assert order.total_price == 50.0
    assert len(order.items) == 1
    assert order.items[0].quantity == 2

    assert CartItem.query.count() == 0
    assert db.session.get(Product, cart_with_item["product"].id).stock_quantity == 3


def test_checkout_redirects_to_orders_on_success(client, cart_with_item):
    _login(client, cart_with_item["customer"])
    response = client.post("/cart/checkout")
    assert response.status_code == 302
    assert response.headers["Location"] == "/orders/"


def test_checkout_insufficient_stock_keeps_cart_intact(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]  # 5 in stock
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product=product, quantity=10)  # more than in stock
    db.session.add(item)
    db.session.commit()
    item_id = item.id

    _login(client, customer)
    response = client.post("/cart/checkout", follow_redirects=True)

    assert b"Not enough stock" in response.data
    assert Order.query.count() == 0
    assert db.session.get(CartItem, item_id) is not None  # cart survives a failed checkout
    assert db.session.get(Product, product.id).stock_quantity == 5


def test_checkout_multi_item_cart_one_bad_item_blocks_the_whole_checkout(client, app_context, sample_data):
    """create_order applies items in one loop with a single commit at the
    end; a failure on the second item must not partially check out the
    first, and the whole cart (including the good item) must survive."""
    customer = sample_data["customer"]
    product = sample_data["product"]  # 5 in stock, fine
    out_of_stock = Product(name="Out of Stock", price=10.0, stock_quantity=0, category=sample_data["category"])
    db.session.add(out_of_stock)
    db.session.flush()

    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    db.session.add_all([
        CartItem(cart=cart, product=product, quantity=1),
        CartItem(cart=cart, product=out_of_stock, quantity=1),
    ])
    db.session.commit()

    _login(client, customer)
    response = client.post("/cart/checkout", follow_redirects=True)

    assert b"Not enough stock" in response.data
    assert Order.query.count() == 0
    assert CartItem.query.count() == 2  # nothing removed from the cart
    assert db.session.get(Product, product.id).stock_quantity == 5  # untouched


def test_checkout_multiple_items(client, app_context, sample_data):
    customer = sample_data["customer"]
    product = sample_data["product"]
    second = Product(name="USB Keyboard", price=40.0, stock_quantity=3, category=sample_data["category"])
    db.session.add(second)
    db.session.flush()

    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    db.session.add_all([
        CartItem(cart=cart, product=product, quantity=1),
        CartItem(cart=cart, product=second, quantity=2),
    ])
    db.session.commit()

    _login(client, customer)
    response = client.post("/cart/checkout")

    assert response.status_code == 302
    order = Order.query.first()
    assert order.total_price == 25.0 + 2 * 40.0
    assert CartItem.query.count() == 0


def test_checkout_does_not_touch_other_customers_cart(client, app_context, sample_data):
    other = _other_customer("other-checkout@example.com")
    other_cart = Cart(user_id=other.id)
    db.session.add(other_cart)
    db.session.flush()
    other_item = CartItem(cart=other_cart, product=sample_data["product"], quantity=1)
    db.session.add(other_item)
    db.session.commit()
    other_item_id = other_item.id

    _login(client, sample_data["customer"])  # logged in as someone with no cart at all
    response = client.post("/cart/checkout", follow_redirects=True)

    assert b"Your cart is empty." in response.data
    assert db.session.get(CartItem, other_item_id) is not None
    assert Order.query.count() == 0


def test_checkout_orphaned_item_blocks_checkout_with_clear_error(client, app_context, sample_data):
    """A cart item pointing at a since-deleted product can't be turned
    into an order; create_order reports it as a normal nonexistent-product
    error rather than crashing, and the cart is left untouched."""
    customer = sample_data["customer"]
    product = sample_data["product"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    item = CartItem(cart=cart, product_id=product.id, quantity=1)
    db.session.add(item)
    db.session.commit()

    db.session.delete(product)
    db.session.commit()

    _login(client, customer)
    response = client.post("/cart/checkout", follow_redirects=True)

    assert response.status_code == 200
    assert b"does not exist" in response.data
    assert Order.query.count() == 0
    assert CartItem.query.count() == 1
