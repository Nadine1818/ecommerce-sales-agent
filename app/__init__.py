import os

from flask import Flask

from app.extensions import db


def create_app():
    app = Flask(__name__)

    # basedir = the project root (ecommerce-sales-agent/), so the SQLite
    # file always resolves to the same place regardless of where you run
    # `python run.py` from.
    basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(basedir, 'data', 'app.db')}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

    # attach the shared db instance to this specific app instance
    db.init_app(app)

    with app.app_context():
        # context is required to create the database tables, because db.Model needs to know which app it's associated with
        # importing models here ensures that all the model classes are registered with SQLAlchemy before calling create_all()
        from app import models  # noqa: F401 — import registers the models with db.Model
        # create the database tables if they don't exist yet
        db.create_all()

    # PLACEHOLDER FOR BLUEPRINT REGISTRATION

    return app