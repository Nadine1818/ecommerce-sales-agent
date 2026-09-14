import pytest
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Order, OrderItem, User


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
