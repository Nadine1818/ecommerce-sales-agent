from flask import Blueprint

orders_bp = Blueprint("orders", __name__, template_folder="templates")

# routes.py needs orders_bp to already exist before it can attach
# @orders_bp.route(...) decorators to it.
from app.orders import routes
