from flask import Blueprint

chat_bp = Blueprint("chat", __name__, template_folder="templates")

# Imported at the bottom, not the top: routes.py does `from app.chat import
# chat_bp` to attach its @chat_bp.route(...) decorators to this exact
# object. If we imported routes.py before chat_bp existed, that import
# would fail, this ordering guarantees chat_bp is fully defined first.
from app.chat import routes  