from app.extensions import db


class Category(db.Model):
    __tablename__ = "categories"

    id = db.Column(db.Integer, primary_key=True)
    # required field, must be unique (no two categories can have the same name)
    name = db.Column(db.String(100), nullable=False, unique=True)

    # category.products will return a list of Product objects that belong to this category.
    products = db.relationship("Product", back_populates="category")

    def __repr__(self):
        return f"<Category {self.id} {self.name}>"

    def to_dict(self):
        # Used by the dashboard and any API responses — keeps serialization
        # logic in one place instead of repeating it in every route.
        return {"id": self.id, "name": self.name}