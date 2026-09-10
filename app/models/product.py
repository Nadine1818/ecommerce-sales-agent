from app.extensions import db


class Product(db.Model):
    __tablename__ = "products"

    # name, price, stock_quantity, and category_id are required fields. description is optional.
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    price = db.Column(db.Float, nullable=False)
    stock_quantity = db.Column(db.Integer, nullable=False, default=0)

    # The foreign key column, linking this product to its category. This is what actually creates the DB column.
    category_id = db.Column(db.Integer, db.ForeignKey("categories.id"), nullable=False)

    # product.catrgory returns the category object that this product belongs to, backpopulates keeps both sides in sync
    category = db.relationship("Category", back_populates="products")

    def __repr__(self):
        return f"<Product {self.id} {self.name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "stock_quantity": self.stock_quantity,
            "category_id": self.category_id,
        }