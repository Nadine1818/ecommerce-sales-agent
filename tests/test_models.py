import pytest
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Cart, CartItem, Category, Order, OrderItem, Product, User


def test_deleting_order_deletes_its_items(sample_data):
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=25.0)
    order.items.append(
        OrderItem(product=sample_data["product"], quantity=1, price_at_order=25.0)
    )
    db.session.add(order)
    db.session.commit()
    order_item_id = order.items[0].id

    db.session.delete(order)
    db.session.commit()

    assert db.session.get(OrderItem, order_item_id) is None


def test_customer_email_must_be_unique(sample_data):
    db.session.add(
        User(name="Someone Else", email=sample_data["customer"].email, password_hash="test")
    )

    with pytest.raises(IntegrityError):
        db.session.commit()


def test_category_name_must_be_unique(sample_data):
    db.session.add(Category(name=sample_data["category"].name))

    with pytest.raises(IntegrityError):
        db.session.commit()


def test_deleting_user_deletes_their_cart_and_cart_items(sample_data):
    """User.cart uses cascade="all, delete-orphan" — a deleted account
    must not leave an orphaned cart or cart_item row behind."""
    customer = sample_data["customer"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    db.session.add(CartItem(cart=cart, product=sample_data["product"], quantity=1))
    db.session.commit()
    cart_id = cart.id

    db.session.delete(customer)
    db.session.commit()

    assert db.session.get(Cart, cart_id) is None
    assert CartItem.query.filter_by(cart_id=cart_id).count() == 0


def test_cart_user_id_must_be_unique_one_cart_per_user(sample_data):
    """Cart.user_id has unique=True — enforced at the DB level, which is
    what add_to_cart's cart-creation race-recovery code (see
    tests/test_tools.py's KNOWN BUG test) actually relies on to detect
    the race in the first place."""
    db.session.add(Cart(user_id=sample_data["customer"].id))
    db.session.commit()

    db.session.add(Cart(user_id=sample_data["customer"].id))
    with pytest.raises(IntegrityError):
        db.session.commit()


def test_product_stock_quantity_defaults_to_zero(sample_data):
    product = Product(name="No Stock Set", price=1.0, category=sample_data["category"])
    db.session.add(product)
    db.session.commit()

    assert product.stock_quantity == 0


def test_order_status_defaults_to_pending(sample_data):
    order = Order(customer_id=sample_data["customer"].id)
    db.session.add(order)
    db.session.commit()

    assert order.status == "pending"


def test_user_role_defaults_to_customer(app_context):
    user = User(name="Plain", email="plain@example.com", password_hash="x")
    db.session.add(user)
    db.session.commit()

    assert user.role == "customer"


def test_user_to_dict_excludes_password_hash(sample_data):
    customer = sample_data["customer"]
    customer.set_password("secret123")
    db.session.commit()

    data = customer.to_dict()

    assert "password_hash" not in data
    assert data == {"id": customer.id, "name": customer.name, "email": customer.email, "role": "customer"}


def test_user_check_password_rejects_wrong_password(sample_data):
    customer = sample_data["customer"]
    customer.set_password("correct-password")
    db.session.commit()

    assert customer.check_password("correct-password") is True
    assert customer.check_password("wrong-password") is False


def test_category_to_dict_shape(sample_data):
    assert sample_data["category"].to_dict() == {"id": sample_data["category"].id, "name": "Electronics"}


def test_product_to_dict_shape(sample_data):
    product = sample_data["product"]
    assert product.to_dict() == {
        "id": product.id,
        "name": "Wireless Mouse",
        "description": None,
        "price": 25.0,
        "stock_quantity": 5,
        "category_id": sample_data["category"].id,
    }


def test_deleting_category_with_products_is_blocked(sample_data):
    """Category<->Product has no cascade configured, so when a Category
    is deleted, SQLAlchemy's default relationship handling tries to null
    out the orphaned products' category_id first — which fails, because
    that column is nullable=False. Net effect: deleting a category that
    still has products raises IntegrityError rather than silently
    orphaning the products or succeeding. Documents the real behavior of
    this relationship (note this is a NOT NULL violation from the ORM's
    own nulling attempt, not a raw SQLite foreign-key check — SQLite's
    own FK enforcement is off by default here, as noted in
    tests/test_cart.py's product-deletion tests)."""
    with pytest.raises(IntegrityError):
        db.session.delete(sample_data["category"])
        db.session.commit()
