"""Admin dashboard routes. Every route is protected with
@login_required(role="admin"), a logged-in customer can't reach any of
these, nor an anonymous visitor"""

from flask import flash, redirect, render_template, request, session, url_for

from app.auth.decorators import login_required
from app.dashboard import dashboard_bp
from app.models import Order, Product, User
from app.rag import add_or_update_item, delete_item, get_all_items

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
            new_id = _next_id("policy")
            add_or_update_item(
                item_id=f"policy-{new_id}",
                item_type="policy",
                data={"id": new_id, "title": title, "content": content},
            )

        else:
            flash("Invalid item type.")
            return render_template("rag_form.html")

        flash("Knowledge base entry added.")
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
        flash("Knowledge base entry updated.")
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
    delete_item(item_id)
    flash("Knowledge base entry deleted.")
    return redirect(url_for("dashboard.rag_data"))