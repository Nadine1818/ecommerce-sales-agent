import os

from dotenv import load_dotenv
from flask import Flask

from app.extensions import db

_project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

def create_app(test_config=None):
    load_dotenv()

    app = Flask(__name__)

    # basedir = the project root (ecommerce-sales-agent/), so the SQLite
    # file always resolves to the same place regardless of where you run
    # `python run.py` from.
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(basedir, 'data', 'app.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

    # lets tests override config (e.g. an in-memory database) without
    # touching the real app.db file.
    if test_config:
        app.config.update(test_config)

    # attach the shared db instance to this specific app instance
    db.init_app(app)

    with app.app_context():
        # context is required to create the database tables, because db.Model needs to know which app it's associated with
        # importing models here ensures that all the model classes are registered with SQLAlchemy before calling create_all()
        from app import models  # noqa: F401 — import registers the models with db.Model
        # create the database tables if they don't exist yet
        db.create_all()

    # Imported here, inside create_app(), for the same reason models are:
    # app.chat.routes imports compiled_graph (from app.agent), which in
    # turn eventually needs the Flask app/db to be set up first.
    from app.chat import chat_bp
    from app.auth import auth_bp
    from app.cart import cart_bp
    from app.orders import orders_bp
    from app.dashboard import dashboard_bp
 
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(cart_bp, url_prefix="/cart")
    app.register_blueprint(orders_bp, url_prefix="/orders")
    app.register_blueprint(dashboard_bp, url_prefix="/dashboard")
    
    return app