from flask import Blueprint

cart_bp = Blueprint("cart", __name__, template_folder="templates")

# routes.py needs cart_bp to already exist before it can attach
# @cart_bp.route(...) decorators to it.
from app.cart import routes
