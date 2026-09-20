"""Admin dashboard routes. Every route is protected with
@login_required(role="admin"), a logged-in customer can't reach any of
these, nor an anonymous visitor"""

import re
import threading

from flask import abort, flash, redirect, render_template, request, session, url_for

from app.auth.decorators import login_required
from app.dashboard import dashboard_bp
from app.extensions import db
from app.models import CartItem, Category, Order, OrderItem, Product, User
from app.rag import add_or_update_item, delete_item, get_all_items

# Guards _next_id()'s read-max-then-write-one-higher sequence in rag_add.
# Without it, two near-simultaneous "add" submissions can both read the
# same current max and both compute the same "next" id 
_rag_add_lock = threading.Lock()

# Same format check as auth/routes.py's registration form — kept as its
# own copy rather than importing auth's constant, so this module doesn't
# reach into auth's internals for something this small.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _product_rag_data(product: Product) -> dict:
    # Same shape ingest_knowledge_base.py builds for a product, so a
    # dashboard-created/edited product is embedded identically to one
    # ingested from the bulk script.
    return {
        "id": product.id,
        "name": product.name,
        "description": product.description or "",
        "price": product.price,
        "stock_quantity": product.stock_quantity,
        "category_id": product.category_id,
        "category_name": product.category.name,
    }

# redirects to /dashboard/products, the main dashboard page
@dashboard_bp.route("/")
@login_required(role="admin")
def index():
    return redirect(url_for("dashboard.products"))

# renders the products page, user_name is passed to the template for display
@dashboard_bp.route("/products")
@login_required(role="admin")
def products():
    all_products = Product.query.all()
    return render_template("products.html", user_name=session["name"], products=all_products)


def _validate_product_form(form) -> tuple[dict, list[str]]:
    """Shared by product_add and product_edit. Returns (cleaned_data, errors) —
    cleaned_data has the right types only when errors is empty."""
    errors = []

    name = form.get("name", "").strip()
    description = form.get("description", "").strip()
    price_raw = form.get("price", "").strip()
    stock_raw = form.get("stock_quantity", "").strip()
    category_id_raw = form.get("category_id", "").strip()

    if not name:
        errors.append("Name is required.")

    price = None
    try:
        price = float(price_raw)
        if price < 0:
            errors.append("Price can't be negative.")
    except ValueError:
        errors.append("Price must be a number.")

    stock_quantity = None
    try:
        stock_quantity = int(stock_raw)
        if stock_quantity < 0:
            errors.append("Stock quantity can't be negative.")
    except ValueError:
        errors.append("Stock quantity must be a whole number.")

    category = None
    try:
        category = db.session.get(Category, int(category_id_raw))
    except ValueError:
        category = None
    if category is None:
        errors.append("Please choose a valid category.")

    return (
        {
            "name": name,
            "description": description,
            "price": price,
            "stock_quantity": stock_quantity,
            "category": category,
        },
        errors,
    )


@dashboard_bp.route("/products/add", methods=["GET", "POST"])
@login_required(role="admin")
def product_add():
    categories = Category.query.order_by(Category.name).all()

    if request.method == "POST":
        cleaned, errors = _validate_product_form(request.form)
        if errors:
            for error in errors:
                flash(error)
            return render_template("product_form.html", categories=categories, form=request.form)

        product = Product(
            name=cleaned["name"],
            description=cleaned["description"] or None,
            price=cleaned["price"],
            stock_quantity=cleaned["stock_quantity"],
            category=cleaned["category"],
        )
        db.session.add(product)
        db.session.commit()

        # Embed it immediately, same as RAG add/edit already does for
        # FAQs/policies — no need to wait for ingest_knowledge_base.py
        # to run again for a brand-new product to be searchable.
        add_or_update_item(
            item_id=f"product-{product.id}",
            item_type="product",
            data=_product_rag_data(product),
        )

        flash(f"Added {product.name}.", "success")
        return redirect(url_for("dashboard.products"))

    return render_template("product_form.html", categories=categories, form=None)


@dashboard_bp.route("/products/edit/<int:product_id>", methods=["GET", "POST"])
@login_required(role="admin")
def product_edit(product_id):
    product = db.get_or_404(Product, product_id)
    categories = Category.query.order_by(Category.name).all()

    if request.method == "POST":
        cleaned, errors = _validate_product_form(request.form)
        if errors:
            for error in errors:
                flash(error)
            return render_template(
                "product_form.html", categories=categories, form=request.form, product=product
            )

        product.name = cleaned["name"]
        product.description = cleaned["description"] or None
        product.price = cleaned["price"]
        product.stock_quantity = cleaned["stock_quantity"]
        product.category = cleaned["category"]
        db.session.commit()

        add_or_update_item(
            item_id=f"product-{product.id}",
            item_type="product",
            data=_product_rag_data(product),
        )

        flash(f"Updated {product.name}.", "success")
        return redirect(url_for("dashboard.products"))

    return render_template("product_form.html", categories=categories, form=None, product=product)


@dashboard_bp.route("/products/delete/<int:product_id>", methods=["POST"])
@login_required(role="admin")
def product_delete(product_id):
    product = db.get_or_404(Product, product_id)

    # Deleting a product that's part of real order history would erase
    # part of that history's context (order pages already fall back to
    # "(deleted product)"/"Product #<id>" gracefully, but it's still
    # better to keep the record intact when possible) — block it rather
    # than silently degrade past orders.
    was_ordered = OrderItem.query.filter_by(product_id=product_id).first() is not None
    if was_ordered:
        flash(
            f"Can't delete {product.name} — it appears in existing order history. "
            "Historical orders keep a record of what was actually purchased."
        )
        return redirect(url_for("dashboard.products"))

    # Cart lines aren't historical records the way orders are, so it's
    # fine to just clear any that reference this product rather than
    # leaving them pointing at a product_id that no longer exists.
    CartItem.query.filter_by(product_id=product_id).delete()
    db.session.delete(product)
    db.session.commit()

    delete_item(f"product-{product_id}")

    flash(f"Deleted {product.name}.")
    return redirect(url_for("dashboard.products"))


@dashboard_bp.route("/orders")
@login_required(role="admin")
def orders():
    all_orders = Order.query.order_by(Order.created_at.desc()).all()
    return render_template("admin_orders.html", user_name=session["name"], orders=all_orders)

# role specified to avoid adding the admin to the customer list
@dashboard_bp.route("/customers")
@login_required(role="admin")
def customers():
    all_customers = User.query.filter_by(role="customer").all()
    return render_template("customers.html", user_name=session["name"], customers=all_customers)


def _get_customer_or_404(user_id: int) -> User:
    # Deliberately scoped to role="customer", not just User.id — this
    # route is for managing customer accounts, and must not become a
    # backdoor for loading (and then editing/deleting) an admin account
    # by guessing a different user_id in the URL.
    customer = User.query.filter_by(id=user_id, role="customer").first()
    if customer is None:
        abort(404)
    return customer


@dashboard_bp.route("/customers/add", methods=["GET", "POST"])
@login_required(role="admin")
def customer_add():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        errors = []
        if not name or not email or not password:
            errors.append("Name, email, and password are all required.")
        elif not EMAIL_RE.match(email):
            errors.append("Please enter a valid email address.")
        elif len(password) < MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")
        elif User.query.filter_by(email=email).first():
            errors.append("An account with that email already exists.")

        if errors:
            for error in errors:
                flash(error)
            return render_template("customer_form.html", form=request.form)

        # Always role="customer" — same rule as the public registration
        # form: there is no path, including this one, that creates an
        # admin account outside of seed.py.
        customer = User(name=name, email=email, role="customer")
        customer.set_password(password)
        db.session.add(customer)
        db.session.commit()

        flash(f"Added {customer.name}.", "success")
        return redirect(url_for("dashboard.customers"))

    return render_template("customer_form.html", form=None)


@dashboard_bp.route("/customers/edit/<int:user_id>", methods=["GET", "POST"])
@login_required(role="admin")
def customer_edit(user_id):
    customer = _get_customer_or_404(user_id)

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()

        errors = []
        if not name or not email:
            errors.append("Name and email are required.")
        elif not EMAIL_RE.match(email):
            errors.append("Please enter a valid email address.")
        elif User.query.filter(User.email == email, User.id != customer.id).first():
            errors.append("Another account already uses that email.")

        if errors:
            for error in errors:
                flash(error)
            return render_template("customer_form.html", form=request.form, customer=customer)

        customer.name = name
        customer.email = email
        db.session.commit()

        flash(f"Updated {customer.name}.", "success")
        return redirect(url_for("dashboard.customers"))

    return render_template("customer_form.html", form=None, customer=customer)


@dashboard_bp.route("/customers/delete/<int:user_id>", methods=["POST"])
@login_required(role="admin")
def customer_delete(user_id):
    customer = _get_customer_or_404(user_id)

    # Same reasoning as product deletion: an existing order is a
    # historical record (admin_orders.html already falls back to
    # "Unknown" for a missing customer, but it's better not to erase
    # that context when it's avoidable).
    has_orders = Order.query.filter_by(customer_id=user_id).first() is not None
    if has_orders:
        flash(
            f"Can't delete {customer.name} — they have existing order history. "
            "Historical orders keep a record of who placed them."
        )
        return redirect(url_for("dashboard.customers"))

    # User.cart has cascade="all, delete-orphan", so deleting the user
    # also deletes their Cart and, in turn, its CartItems — no manual
    # cleanup needed here the way product deletion needs for CartItem.
    db.session.delete(customer)
    db.session.commit()

    flash(f"Deleted {customer.name}.")
    return redirect(url_for("dashboard.customers"))


@dashboard_bp.route("/rag")
@login_required(role="admin")
def rag_data():
    faqs = get_all_items(item_type="faq")
    policies = get_all_items(item_type="policy")
    return render_template("rag_data.html", user_name=session["name"], faqs=faqs, policies=policies)


def _next_id(item_type: str) -> int:
    # New entries need a numeric id that doesn't collide with an existing
    # one of the same type, one more than whatever the current highest
    # id is, or 1 if this is the very first entry of that type.
    existing = get_all_items(item_type=item_type)
    if not existing:
        return 1
    return max(int(item["metadata"]["id"]) for item in existing) + 1


@dashboard_bp.route("/rag/add", methods=["GET", "POST"])
@login_required(role="admin")
def rag_add():
    if request.method == "POST":
        item_type = request.form.get("item_type")

        if item_type == "faq":
            question = request.form.get("question", "").strip()
            answer = request.form.get("answer", "").strip()
            if not question or not answer:
                flash("Question and answer are required.")
                return render_template("rag_form.html")
            with _rag_add_lock:
                new_id = _next_id("faq")
                add_or_update_item(
                    item_id=f"faq-{new_id}",
                    item_type="faq",
                    data={"id": new_id, "question": question, "answer": answer},
                )

        elif item_type == "policy":
            title = request.form.get("title", "").strip()
            content = request.form.get("content", "").strip()
            if not title or not content:
                flash("Title and content are required.")
                return render_template("rag_form.html")
            with _rag_add_lock:
                new_id = _next_id("policy")
                add_or_update_item(
                    item_id=f"policy-{new_id}",
                    item_type="policy",
                    data={"id": new_id, "title": title, "content": content},
                )

        else:
            flash("Invalid item type.")
            return render_template("rag_form.html")

        flash("Knowledge base entry added.", "success")
        return redirect(url_for("dashboard.rag_data"))

    return render_template("rag_form.html")


@dashboard_bp.route("/rag/edit/<item_id>", methods=["GET", "POST"])
@login_required(role="admin")
def rag_edit(item_id):
    # item_id looks like "faq-3" or "policy-1", split once to recover
    # the type and the numeric id ingest.py's add_or_update_item needs.
    item_type, numeric_id = item_id.split("-", 1)

    if request.method == "POST":
        if item_type == "faq":
            question = request.form.get("question", "").strip()
            answer = request.form.get("answer", "").strip()
            add_or_update_item(
                item_id=item_id,
                item_type="faq",
                data={"id": numeric_id, "question": question, "answer": answer},
            )
        elif item_type == "policy":
            title = request.form.get("title", "").strip()
            content = request.form.get("content", "").strip()
            add_or_update_item(
                item_id=item_id,
                item_type="policy",
                data={"id": numeric_id, "title": title, "content": content},
            )
        flash("Knowledge base entry updated.", "success")
        return redirect(url_for("dashboard.rag_data"))

    # For GET requests, we need to fetch the existing item data to pre-fill the form.
    # gets the list of all items of the given type, then finds the one with the matching id
    # so the form can be pre-filled with the item's existing data for editing.
    existing = get_all_items(item_type=item_type)
    item = next((i for i in existing if i["id"] == item_id), None)
    if item is None:
        flash("Item not found.")
        return redirect(url_for("dashboard.rag_data"))

    return render_template("rag_form.html", item=item, item_type=item_type)

# used only post method as no form is rendered for deletion, just a button click that triggers the POST request
# delete has to come only from an actual post request, not a GET request, to avoid accidental deletions from link clicks or page refreshes.
@dashboard_bp.route("/rag/delete/<item_id>", methods=["POST"])
@login_required(role="admin")
def rag_delete(item_id):
    deleted = delete_item(item_id)
    if deleted:
        flash("Knowledge base entry deleted.")
    else:
        flash("That entry no longer exists , nothing to delete.")
    return redirect(url_for("dashboard.rag_data"))