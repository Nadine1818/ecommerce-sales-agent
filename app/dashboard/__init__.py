from flask import Blueprint

dashboard_bp = Blueprint("dashboard", __name__, template_folder="templates")

# routes.py needs dashboard_bp to already exist before it can attach
# @dashboard_bp.route(...) decorators to it.
from app.dashboard import routes