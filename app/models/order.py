from datetime import datetime

from app.extensions import db


class Order(db.Model):
    __tablename__ = "orders"

    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey("customers.id"), nullable=False)
    status = db.Column(db.String(50), nullable=False, default="pending")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    total_price = db.Column(db.Float, nullable=False, default=0.0)
    # order.customer returns the customer object that placed this order, backpopulates keeps both sides in sync
    customer = db.relationship("Customer", back_populates="orders")
    # order.items returns a list of OrderItem objects that belong to this order, backpopulates keeps both sides in sync
    # An order has many items. cascade="all, delete-orphan" deletes all associated OrderItem rows if the Order is deleted, and prevents orphaned OrderItem rows.
    items = db.relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Order {self.id} status={self.status}>"

    def to_dict(self):
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "total_price": self.total_price,
            "items": [item.to_dict() for item in self.items],
        }


