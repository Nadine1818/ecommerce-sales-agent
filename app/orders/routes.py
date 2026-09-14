"""Routes for the customer's order history page. It is Read-only since orders are
only ever created by cart checkout or the chat agent's create_order tool."""

from flask import render_template, session

from app.auth.decorators import login_required
from app.models import Order
from app.orders import orders_bp


@orders_bp.route("/")
@login_required(role="customer")
def index():
    orders = (
        Order.query.filter_by(customer_id=session["user_id"])
        .order_by(Order.created_at.desc())
        .all()
    )
    return render_template("orders.html", user_name=session["name"], orders=orders)
