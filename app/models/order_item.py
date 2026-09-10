from app.extensions import db


class OrderItem(db.Model):
    __tablename__ = "order_items"

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("orders.id"), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)

    # Snapshot of the product's price at the moment this order was placed
    # we deliberately don't rely on product.price here because if the product's
    # price changes later, past orders must still reflect what the customer
    # actually paid
    price_at_order = db.Column(db.Float, nullable=False)
    # orderitem.order returns the order object that this order item belongs to, backpopulates keeps both sides in sync
    order = db.relationship("Order", back_populates="items")
    # orderitem.product returns the product object that this order item belongs to, no back_populates needed because we don't need to access order items from the product side
    product = db.relationship("Product")

    def __repr__(self):
        return f"<OrderItem order={self.order_id} product={self.product_id} qty={self.quantity}>"

    def to_dict(self):
        return {
            "id": self.id,
            "order_id": self.order_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "price_at_order": self.price_at_order,
        }