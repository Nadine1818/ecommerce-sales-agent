from flask_sqlalchemy import SQLAlchemy

# Single shared db instance.
# app/__init__.py calls db.init_app(app) inside create_app().
# Every model file imports `db` from here, never from app/__init__.py directly —
# that would create a circular import (app imports models, models would import app).
db = SQLAlchemy()