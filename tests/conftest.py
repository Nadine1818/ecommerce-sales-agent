import pytest
from sqlalchemy.pool import StaticPool

from app import create_app
from app.extensions import db
from app.models import Category, Product, User


@pytest.fixture()
def app():
    """A Flask app wired to a fresh in-memory SQLite database, so tests
    never touch the real data/app.db file. StaticPool is required here —
    without it, SQLite's :memory: database is per-connection, and
    Flask-SQLAlchemy would hand out a different (empty) database on
    every connection it opens."""
    application = create_app(
        test_config={
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SQLALCHEMY_ENGINE_OPTIONS": {"poolclass": StaticPool},
            "TESTING": True,
        }
    )
    yield application


@pytest.fixture()
def app_context(app):
    with app.app_context():
        yield app


@pytest.fixture()
def sample_data(app_context):
    """A category, a customer, and a product with 5 units in stock —
    enough for the create_order tests to exercise both the happy path
    and the stock/existence error paths."""
    category = Category(name="Electronics")
    customer = User(name="Test Customer", email="test@example.com", password_hash="test")
    product = Product(name="Wireless Mouse", price=25.0, stock_quantity=5, category=category)

    db.session.add_all([category, customer, product])
    db.session.commit()

    return {"category": category, "customer": customer, "product": product}
