from app.extensions import db


class CartItem(db.Model):
    __tablename__ = "cart_items"

    id = db.Column(db.Integer, primary_key=True)
    cart_id = db.Column(db.Integer, db.ForeignKey("carts.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    # do cart_item.cart to see the cart this item belongs to
    cart = db.relationship("Cart", back_populates="items")
    # One-directional, do cart_item.product to see the product this item represents. 
    # No back_populates needed since we don't need to see "all cart items for a product" anywhere.
    product = db.relationship("Product")

    def __repr__(self):
        return f"<CartItem cart={self.cart_id} product={self.product_id} qty={self.quantity}>"

    def to_dict(self):
        return {
            "id": self.id,
            "cart_id": self.cart_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
        }