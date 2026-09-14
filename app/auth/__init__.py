from flask import Blueprint

auth_bp = Blueprint("auth", __name__, template_folder="templates")

# routes.py needs auth_bp to already
# exist before it can attach @auth_bp.route(...) decorators to it.
from app.auth import routes 