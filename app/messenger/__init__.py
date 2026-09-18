from flask import Blueprint

messenger_bp = Blueprint("messenger", __name__)

from app.messenger import routes