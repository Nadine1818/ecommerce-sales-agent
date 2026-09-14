"""Routes for the customer's shopping cart page to view items, adjust
quantities, remove items, and check out. Checkout hands the cart's items
to the same create_order tool the chat agent uses, so a cart checkout and
an agent-placed order go through identical stock/price logic."""

from flask import abort, flash, redirect, render_template, request, session, url_for

from app.agent.tools import create_order
from app.auth.decorators import login_required
from app.cart import cart_bp
from app.extensions import db
from app.models import Cart, CartItem


@cart_bp.route("/")
@login_required(role="customer")
def index():
    cart = Cart.query.filter_by(user_id=session["user_id"]).first()
    items = cart.items if cart else []
    # a cart item's product can be gone if it was removed from the catalog
    # after being added to a cart; it contributes nothing to the total
    total = sum(item.product.price * item.quantity for item in items if item.product)
    return render_template("cart.html", user_name=session["name"], items=items, total=total)


@cart_bp.route("/update/<int:item_id>", methods=["POST"])
@login_required(role="customer")
def update(item_id):
    item = db.get_or_404(CartItem, item_id)
    # a customer should only ever be able to edit their own cart items
    if item.cart.user_id != session["user_id"]:
        abort(404)

    if item.product is None:
        flash("That product is no longer available.")
        return redirect(url_for("cart.index"))

    quantity = request.form.get("quantity", type=int)
    if quantity is None or quantity < 1:
        flash("Quantity must be at least 1.")
    elif quantity > item.product.stock_quantity:
        flash(f"Only {item.product.stock_quantity} of {item.product.name} left in stock.")
    else:
        item.quantity = quantity
        db.session.commit()

    return redirect(url_for("cart.index"))


@cart_bp.route("/remove/<int:item_id>", methods=["POST"])
@login_required(role="customer")
def remove(item_id):
    item = db.get_or_404(CartItem, item_id)
    if item.cart.user_id != session["user_id"]:
        abort(404)

    product_name = item.product.name if item.product else "item"
    db.session.delete(item)
    db.session.commit()
    flash(f"Removed {product_name} from your cart.")
    return redirect(url_for("cart.index"))


@cart_bp.route("/checkout", methods=["POST"])
@login_required(role="customer")
def checkout():
    customer_id = session["user_id"]
    cart = Cart.query.filter_by(user_id=customer_id).first()
    if cart is None or not cart.items:
        flash("Your cart is empty.")
        return redirect(url_for("cart.index"))

    items = [{"product_id": item.product_id, "quantity": item.quantity} for item in cart.items]
    result = create_order.invoke({"customer_id": customer_id, "items": items})

    if "error" in result:
        flash(result["error"])
        return redirect(url_for("cart.index"))

    # removes all items from the cart after checkout
    for item in list(cart.items):
        db.session.delete(item)
    db.session.commit()

    flash(f"Order #{result['order_id']} placed — total ${result['total_price']:.2f}.")
    return redirect(url_for("orders.index"))
