"""Tests for the admin dashboard (app/dashboard/routes.py): role-based
access control, the Products/Orders/Customers read views, and full CRUD
on RAG knowledge-base entries (FAQs and policies).

The RAG storage layer (add_or_update_item/get_all_items/delete_item) is
replaced with an in-memory FakeRagStore rather than hitting the real
ChromaDB/embedding pipeline — that pipeline's own behavior is already
covered by tests/test_rag.py. This file is about the ROUTE logic:
form validation, id assignment, redirects, flashes, and auth gating."""

import threading
import time

import pytest

from app.dashboard import routes as dashboard_routes
from app.extensions import db
from app.models import Order, OrderItem, User


class FakeRagStore:
    """In-memory stand-in for the app.rag module functions dashboard
    routes call, keyed exactly like the real ChromaDB-backed one:
    item_id -> {"id": item_id, "text": "...", "metadata": {...}}."""

    def __init__(self):
        self.items = {}

    def add_or_update_item(self, item_id, item_type, data):
        metadata = {"type": item_type, **{k: str(v) for k, v in data.items()}}
        self.items[item_id] = {"id": item_id, "text": str(data), "metadata": metadata}

    def delete_item(self, item_id):
        return self.items.pop(item_id, None) is not None

    def get_all_items(self, item_type=None):
        values = list(self.items.values())
        if item_type is None:
            return values
        return [v for v in values if v["metadata"]["type"] == item_type]


@pytest.fixture()
def fake_rag(monkeypatch):
    store = FakeRagStore()
    monkeypatch.setattr(dashboard_routes, "add_or_update_item", store.add_or_update_item)
    monkeypatch.setattr(dashboard_routes, "delete_item", store.delete_item)
    monkeypatch.setattr(dashboard_routes, "get_all_items", store.get_all_items)
    return store


@pytest.fixture()
def client(app):
    return app.test_client()


def _login(client, user):
    with client.session_transaction() as sess:
        sess["user_id"] = user.id
        sess["role"] = user.role
        sess["name"] = user.name


@pytest.fixture()
def admin(app_context):
    admin = User(name="Root", email="admin-dash@example.com", role="admin")
    admin.set_password("adminpw")
    db.session.add(admin)
    db.session.commit()
    return admin


# --------------------------------------------------------------- auth gate

@pytest.mark.parametrize(
    "path",
    ["/dashboard/", "/dashboard/products", "/dashboard/orders", "/dashboard/customers", "/dashboard/rag"],
)
def test_dashboard_routes_require_login(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


@pytest.mark.parametrize(
    "path",
    ["/dashboard/", "/dashboard/products", "/dashboard/orders", "/dashboard/customers", "/dashboard/rag"],
)
def test_dashboard_routes_reject_customer_role(client, path, sample_data):
    _login(client, sample_data["customer"])
    response = client.get(path)
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_dashboard_index_redirects_to_products(client, admin):
    _login(client, admin)
    response = client.get("/dashboard/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/dashboard/products"


# ------------------------------------------------------------- products view

def test_products_page_lists_products(client, admin, app_context, sample_data):
    _login(client, admin)
    response = client.get("/dashboard/products")

    assert response.status_code == 200
    assert b"Wireless Mouse" in response.data


def test_products_page_empty_state(client, admin):
    _login(client, admin)
    response = client.get("/dashboard/products")

    assert b"No products yet." in response.data


def test_products_page_flags_out_of_stock(client, admin, app_context, sample_data):
    sample_data["product"].stock_quantity = 0
    db.session.commit()
    _login(client, admin)

    response = client.get("/dashboard/products")

    assert b"out of stock" in response.data


# --------------------------------------------------------------- orders view

def test_orders_page_lists_all_customers_orders(client, admin, app_context, sample_data):
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=25.0)
    order.items.append(OrderItem(product=sample_data["product"], quantity=1, price_at_order=25.0))
    db.session.add(order)
    db.session.commit()

    _login(client, admin)
    response = client.get("/dashboard/orders")

    assert response.status_code == 200
    assert f"#{order.id}".encode() in response.data
    assert b"Wireless Mouse" in response.data


def test_orders_page_shows_deleted_product_fallback(client, admin, app_context, sample_data):
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=25.0)
    order.items.append(OrderItem(product=sample_data["product"], quantity=1, price_at_order=25.0))
    db.session.add(order)
    db.session.commit()
    db.session.delete(sample_data["product"])
    db.session.commit()

    _login(client, admin)
    response = client.get("/dashboard/orders")

    assert response.status_code == 200
    assert b"(deleted product)" in response.data


def test_orders_page_empty_state(client, admin):
    _login(client, admin)
    response = client.get("/dashboard/orders")
    assert b"No orders yet." in response.data


# ------------------------------------------------------------ customers view

def test_customers_page_lists_customers_only_not_admins(client, admin, app_context, sample_data):
    _login(client, admin)
    response = client.get("/dashboard/customers")

    assert response.status_code == 200
    assert b"Test Customer" in response.data
    assert b"admin-dash@example.com" not in response.data


def test_customers_page_shows_order_count(client, admin, app_context, sample_data):
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=25.0)
    db.session.add(order)
    db.session.commit()

    _login(client, admin)
    response = client.get("/dashboard/customers")

    assert response.status_code == 200
    assert b"1" in response.data  # order count column


# --------------------------------------------------------------- RAG: view

def test_rag_data_page_lists_faqs_and_policies(client, admin, fake_rag):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Q1?", "answer": "A1."})
    fake_rag.add_or_update_item("policy-1", "policy", {"id": 1, "title": "P1", "content": "C1."})

    _login(client, admin)
    response = client.get("/dashboard/rag")

    assert response.status_code == 200
    assert b"Q1?" in response.data
    assert b"P1" in response.data


def test_rag_data_page_empty_state(client, admin, fake_rag):
    _login(client, admin)
    response = client.get("/dashboard/rag")

    assert b"No FAQs yet." in response.data
    assert b"No policies yet." in response.data


# ---------------------------------------------------------------- RAG: add

def test_rag_add_faq_creates_entry_with_next_id(client, admin, fake_rag):
    _login(client, admin)
    response = client.post(
        "/dashboard/rag/add",
        data={"item_type": "faq", "question": "New question?", "answer": "New answer."},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Knowledge base entry added." in response.data
    assert fake_rag.items["faq-1"]["metadata"]["question"] == "New question?"


def test_rag_add_faq_next_id_increments_past_existing(client, admin, fake_rag):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Existing", "answer": "A"})
    fake_rag.add_or_update_item("faq-5", "faq", {"id": 5, "question": "Existing 5", "answer": "A"})

    _login(client, admin)
    client.post("/dashboard/rag/add", data={"item_type": "faq", "question": "New", "answer": "A"})

    assert "faq-6" in fake_rag.items


def test_rag_add_policy_creates_entry(client, admin, fake_rag):
    _login(client, admin)
    response = client.post(
        "/dashboard/rag/add",
        data={"item_type": "policy", "title": "Shipping", "content": "Ships in 2 days."},
        follow_redirects=True,
    )

    assert b"Knowledge base entry added." in response.data
    assert fake_rag.items["policy-1"]["metadata"]["title"] == "Shipping"


@pytest.mark.parametrize(
    "data",
    [
        {"item_type": "faq", "question": "", "answer": "A"},
        {"item_type": "faq", "question": "Q", "answer": ""},
        {"item_type": "faq"},
    ],
)
def test_rag_add_faq_missing_fields_rejected(client, admin, fake_rag, data):
    _login(client, admin)
    response = client.post("/dashboard/rag/add", data=data)

    assert response.status_code == 200
    assert b"Question and answer are required." in response.data
    assert fake_rag.items == {}


@pytest.mark.parametrize(
    "data",
    [
        {"item_type": "policy", "title": "", "content": "C"},
        {"item_type": "policy", "title": "T", "content": ""},
    ],
)
def test_rag_add_policy_missing_fields_rejected(client, admin, fake_rag, data):
    _login(client, admin)
    response = client.post("/dashboard/rag/add", data=data)

    assert response.status_code == 200
    assert b"Title and content are required." in response.data
    assert fake_rag.items == {}


def test_rag_add_invalid_item_type_rejected(client, admin, fake_rag):
    _login(client, admin)
    response = client.post("/dashboard/rag/add", data={"item_type": "not_a_type"})

    assert response.status_code == 200
    assert b"Invalid item type." in response.data
    assert fake_rag.items == {}


def test_rag_add_concurrent_submissions_do_not_collide_on_id(client, admin, fake_rag, monkeypatch):
    """Regression test for a previously-racy _next_id(): two
    near-simultaneous "add FAQ" submissions used to both read the same
    current max id and both write the same "next" id, silently
    overwriting one with the other (see app/dashboard/routes.py's
    _rag_add_lock and its comment). get_all_items (called inside
    _next_id) is slowed down here to force real thread overlap — without
    _rag_add_lock serializing the critical section, this reliably
    reproduces the collision; with it, both submissions must survive
    under distinct ids."""
    _login(client, admin)

    events = []
    events_guard = threading.Lock()
    real_get_all_items = fake_rag.get_all_items

    def slow_get_all_items(item_type=None):
        with events_guard:
            events.append("enter")
        time.sleep(0.05)  # widen the race window _rag_add_lock must close
        result = real_get_all_items(item_type)
        with events_guard:
            events.append("exit")
        return result

    monkeypatch.setattr(dashboard_routes, "get_all_items", slow_get_all_items)

    def add_faq(label):
        client.post(
            "/dashboard/rag/add",
            data={"item_type": "faq", "question": f"Race Q {label}", "answer": f"Race A {label}"},
        )

    t1 = threading.Thread(target=add_faq, args=("A",))
    t2 = threading.Thread(target=add_faq, args=("B",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    # The critical section is serialized: an "exit" always immediately
    # follows its own "enter" — ["enter", "enter", "exit", "exit"] would
    # mean both threads were inside _next_id at once, i.e. the lock failed.
    assert events == ["enter", "exit", "enter", "exit"]

    questions = {f["metadata"]["question"] for f in real_get_all_items(item_type="faq")}
    assert questions == {"Race Q A", "Race Q B"}  # both survived, neither overwrote the other


def test_rag_add_requires_admin(client, fake_rag, sample_data):
    _login(client, sample_data["customer"])
    response = client.post("/dashboard/rag/add", data={"item_type": "faq", "question": "Q", "answer": "A"})

    assert response.status_code == 302
    assert fake_rag.items == {}


# --------------------------------------------------------------- RAG: edit

def test_rag_edit_faq_get_prefills_form(client, admin, fake_rag):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Old Q?", "answer": "Old A."})

    _login(client, admin)
    response = client.get("/dashboard/rag/edit/faq-1")

    assert response.status_code == 200
    assert b"Old Q?" in response.data
    assert b"Old A." in response.data


def test_rag_edit_faq_post_updates_entry_in_place(client, admin, fake_rag):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Old Q?", "answer": "Old A."})

    _login(client, admin)
    response = client.post(
        "/dashboard/rag/edit/faq-1",
        data={"question": "New Q?", "answer": "New A."},
        follow_redirects=True,
    )

    assert b"Knowledge base entry updated." in response.data
    assert fake_rag.items["faq-1"]["metadata"]["question"] == "New Q?"
    assert len(fake_rag.items) == 1  # updated in place, not duplicated


def test_rag_edit_policy_post_updates_entry(client, admin, fake_rag):
    fake_rag.add_or_update_item("policy-1", "policy", {"id": 1, "title": "Old", "content": "Old content"})

    _login(client, admin)
    client.post("/dashboard/rag/edit/policy-1", data={"title": "New", "content": "New content"})

    assert fake_rag.items["policy-1"]["metadata"]["title"] == "New"


def test_rag_edit_nonexistent_item_redirects_with_flash(client, admin, fake_rag):
    _login(client, admin)
    response = client.get("/dashboard/rag/edit/faq-999", follow_redirects=True)

    assert response.status_code == 200
    assert b"Item not found." in response.data


def test_rag_edit_requires_admin(client, fake_rag, sample_data):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Q", "answer": "A"})
    _login(client, sample_data["customer"])

    response = client.get("/dashboard/rag/edit/faq-1")

    assert response.status_code == 302


# ------------------------------------------------------------- RAG: delete

def test_rag_delete_existing_entry(client, admin, fake_rag):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Q", "answer": "A"})

    _login(client, admin)
    response = client.post("/dashboard/rag/delete/faq-1", follow_redirects=True)

    assert b"Knowledge base entry deleted." in response.data
    assert "faq-1" not in fake_rag.items


def test_rag_delete_nonexistent_entry_reports_nothing_to_delete(client, admin, fake_rag):
    _login(client, admin)
    response = client.post("/dashboard/rag/delete/faq-999", follow_redirects=True)

    assert b"nothing to delete" in response.data


def test_rag_delete_requires_admin(client, fake_rag, sample_data):
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Q", "answer": "A"})
    _login(client, sample_data["customer"])

    response = client.post("/dashboard/rag/delete/faq-1")

    assert response.status_code == 302
    assert "faq-1" in fake_rag.items  # untouched


def test_rag_delete_requires_post_not_get(client, admin, fake_rag):
    """No form is ever rendered for delete — it must only be reachable
    via POST, so a stray GET (e.g. a crawler, a refreshed link) can't
    accidentally delete anything."""
    fake_rag.add_or_update_item("faq-1", "faq", {"id": 1, "question": "Q", "answer": "A"})
    _login(client, admin)

    response = client.get("/dashboard/rag/delete/faq-1")

    assert response.status_code == 405
    assert "faq-1" in fake_rag.items


# ------------------------------------------------------------ products: add

def test_product_add_get_renders_form(client, admin, app_context, sample_data):
    _login(client, admin)
    response = client.get("/dashboard/products/add")

    assert response.status_code == 200
    assert b"Add Product" in response.data


def test_product_add_creates_product_and_syncs_rag(client, admin, app_context, sample_data, fake_rag):
    _login(client, admin)
    response = client.post(
        "/dashboard/products/add",
        data={
            "name": "Desk Lamp",
            "description": "LED desk lamp",
            "price": "24.99",
            "stock_quantity": "10",
            "category_id": str(sample_data["category"].id),
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Added Desk Lamp." in response.data
    assert b"Desk Lamp" in response.data  # now showing in the products list

    from app.models import Product
    product = Product.query.filter_by(name="Desk Lamp").first()
    assert product is not None
    assert product.price == 24.99
    assert product.stock_quantity == 10
    # Embedded into the (fake) RAG store immediately, no separate
    # ingestion step required.
    assert fake_rag.items[f"product-{product.id}"]["metadata"]["name"] == "Desk Lamp"


@pytest.mark.parametrize(
    "overrides,expected_error",
    [
        ({"name": ""}, b"Name is required."),
        ({"price": "not-a-number"}, b"Price must be a number."),
        ({"price": "-5"}, b"Price can&#39;t be negative."),
        ({"stock_quantity": "not-a-number"}, b"Stock quantity must be a whole number."),
        ({"stock_quantity": "-1"}, b"Stock quantity can&#39;t be negative."),
        ({"category_id": "99999"}, b"Please choose a valid category."),
    ],
)
def test_product_add_validation_errors(client, admin, app_context, sample_data, fake_rag, overrides, expected_error):
    data = {
        "name": "Desk Lamp",
        "description": "",
        "price": "24.99",
        "stock_quantity": "10",
        "category_id": str(sample_data["category"].id),
    }
    data.update(overrides)

    _login(client, admin)
    response = client.post("/dashboard/products/add", data=data)

    assert response.status_code == 200
    assert expected_error in response.data
    assert fake_rag.items == {}

    from app.models import Product
    assert Product.query.filter_by(name="Desk Lamp").first() is None


def test_product_add_requires_admin(client, sample_data, fake_rag):
    _login(client, sample_data["customer"])
    response = client.post(
        "/dashboard/products/add",
        data={"name": "X", "price": "1", "stock_quantity": "1", "category_id": str(sample_data["category"].id)},
    )

    assert response.status_code == 302
    from app.models import Product
    assert Product.query.filter_by(name="X").first() is None


# ----------------------------------------------------------- products: edit

def test_product_edit_get_prefills_form(client, admin, app_context, sample_data):
    _login(client, admin)
    response = client.get(f"/dashboard/products/edit/{sample_data['product'].id}")

    assert response.status_code == 200
    assert b"Wireless Mouse" in response.data


def test_product_edit_post_updates_in_place_and_resyncs_rag(client, admin, app_context, sample_data, fake_rag):
    product = sample_data["product"]
    _login(client, admin)
    response = client.post(
        f"/dashboard/products/edit/{product.id}",
        data={
            "name": "Wireless Mouse Pro",
            "description": "Updated",
            "price": "29.99",
            "stock_quantity": "3",
            "category_id": str(sample_data["category"].id),
        },
        follow_redirects=True,
    )

    assert b"Updated Wireless Mouse Pro." in response.data

    from app.models import Product
    refreshed = db.session.get(Product, product.id)
    assert refreshed.name == "Wireless Mouse Pro"
    assert refreshed.price == 29.99
    assert fake_rag.items[f"product-{product.id}"]["metadata"]["name"] == "Wireless Mouse Pro"


def test_product_edit_nonexistent_product_404s(client, admin, app_context):
    _login(client, admin)
    response = client.get("/dashboard/products/edit/99999")
    assert response.status_code == 404


# --------------------------------------------------------- products: delete

def test_product_delete_removes_product_and_rag_entry(client, admin, app_context, sample_data, fake_rag):
    product = sample_data["product"]
    fake_rag.add_or_update_item(f"product-{product.id}", "product", {"id": product.id, "name": product.name})

    _login(client, admin)
    response = client.post(f"/dashboard/products/delete/{product.id}", follow_redirects=True)

    assert b"Deleted Wireless Mouse." in response.data
    from app.models import Product
    assert db.session.get(Product, product.id) is None
    assert f"product-{product.id}" not in fake_rag.items


def test_product_delete_blocked_when_product_has_order_history(client, admin, app_context, sample_data, fake_rag):
    product = sample_data["product"]
    order = Order(customer_id=sample_data["customer"].id, status="confirmed", total_price=25.0)
    order.items.append(OrderItem(product=product, quantity=1, price_at_order=25.0))
    db.session.add(order)
    db.session.commit()

    _login(client, admin)
    response = client.post(f"/dashboard/products/delete/{product.id}", follow_redirects=True)

    assert b"existing order history" in response.data
    from app.models import Product
    assert db.session.get(Product, product.id) is not None  # untouched


def test_product_delete_clears_matching_cart_items(client, admin, app_context, sample_data, fake_rag):
    from app.models import Cart, CartItem

    product = sample_data["product"]
    cart = Cart(user_id=sample_data["customer"].id)
    db.session.add(cart)
    db.session.flush()
    db.session.add(CartItem(cart=cart, product=product, quantity=2))
    db.session.commit()

    _login(client, admin)
    client.post(f"/dashboard/products/delete/{product.id}")

    assert CartItem.query.filter_by(product_id=product.id).first() is None


def test_product_delete_requires_admin(client, sample_data, fake_rag):
    _login(client, sample_data["customer"])
    response = client.post(f"/dashboard/products/delete/{sample_data['product'].id}")

    assert response.status_code == 302
    from app.models import Product
    assert db.session.get(Product, sample_data["product"].id) is not None


# ----------------------------------------------------------- customers: add

def test_customer_add_creates_customer(client, admin, app_context):
    _login(client, admin)
    response = client.post(
        "/dashboard/customers/add",
        data={"name": "New Person", "email": "newperson@example.com", "password": "12345678"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Added New Person." in response.data

    from app.models import User
    created = User.query.filter_by(email="newperson@example.com").first()
    assert created is not None
    assert created.role == "customer"
    assert created.check_password("12345678")


@pytest.mark.parametrize(
    "overrides,expected_error",
    [
        ({"name": ""}, b"Name, email, and password are all required."),
        ({"email": "not-an-email"}, b"Please enter a valid email address."),
        ({"password": "short"}, b"Password must be at least 8 characters long."),
    ],
)
def test_customer_add_validation_errors(client, admin, app_context, overrides, expected_error):
    data = {"name": "New Person", "email": "newperson@example.com", "password": "12345678"}
    data.update(overrides)

    _login(client, admin)
    response = client.post("/dashboard/customers/add", data=data)

    assert response.status_code == 200
    assert expected_error in response.data


def test_customer_add_duplicate_email_rejected(client, admin, app_context, sample_data):
    _login(client, admin)
    response = client.post(
        "/dashboard/customers/add",
        data={"name": "Dup", "email": sample_data["customer"].email, "password": "12345678"},
    )

    assert b"An account with that email already exists." in response.data


def test_customer_add_requires_admin(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.post(
        "/dashboard/customers/add",
        data={"name": "X", "email": "x@example.com", "password": "12345678"},
    )

    assert response.status_code == 302
    from app.models import User
    assert User.query.filter_by(email="x@example.com").first() is None


# ---------------------------------------------------------- customers: edit

def test_customer_edit_updates_name_and_email(client, admin, app_context, sample_data):
    customer = sample_data["customer"]
    _login(client, admin)
    response = client.post(
        f"/dashboard/customers/edit/{customer.id}",
        data={"name": "Updated Name", "email": "updated@example.com"},
        follow_redirects=True,
    )

    assert b"Updated Updated Name." in response.data
    refreshed = db.session.get(User, customer.id)
    assert refreshed.name == "Updated Name"
    assert refreshed.email == "updated@example.com"


def test_customer_edit_cannot_target_an_admin_account(client, admin, app_context):
    other_admin = User(name="Other Admin", email="other-admin@example.com", role="admin")
    other_admin.set_password("adminpw")
    db.session.add(other_admin)
    db.session.commit()

    _login(client, admin)
    response = client.get(f"/dashboard/customers/edit/{other_admin.id}")

    assert response.status_code == 404


# -------------------------------------------------------- customers: delete

def test_customer_delete_removes_customer_and_cascades_cart(client, admin, app_context, sample_data):
    from app.models import Cart, CartItem

    customer = sample_data["customer"]
    cart = Cart(user_id=customer.id)
    db.session.add(cart)
    db.session.flush()
    db.session.add(CartItem(cart=cart, product=sample_data["product"], quantity=1))
    db.session.commit()

    _login(client, admin)
    response = client.post(f"/dashboard/customers/delete/{customer.id}", follow_redirects=True)

    assert b"Deleted Test Customer." in response.data
    assert db.session.get(User, customer.id) is None
    assert Cart.query.filter_by(user_id=customer.id).first() is None


def test_customer_delete_blocked_when_customer_has_orders(client, admin, app_context, sample_data):
    customer = sample_data["customer"]
    order = Order(customer_id=customer.id, status="confirmed", total_price=25.0)
    db.session.add(order)
    db.session.commit()

    _login(client, admin)
    response = client.post(f"/dashboard/customers/delete/{customer.id}", follow_redirects=True)

    assert b"existing order history" in response.data
    assert db.session.get(User, customer.id) is not None


def test_customer_delete_requires_admin(client, sample_data):
    _login(client, sample_data["customer"])
    response = client.post(f"/dashboard/customers/delete/{sample_data['customer'].id}")

    assert response.status_code == 302
    assert db.session.get(User, sample_data["customer"].id) is not None