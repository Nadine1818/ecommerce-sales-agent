from datetime import datetime

from app.extensions import db


class Cart(db.Model):
    __tablename__ = "carts"

    id = db.Column(db.Integer, primary_key=True)

    # unique=True enforces "one cart per user" at the database level 
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # user.cart will return the cart for this user, if it exists
    user = db.relationship("User", back_populates="cart")
    # do cart.items to see the items in the cart
    items = db.relationship("CartItem", back_populates="cart", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Cart {self.id} user={self.user_id}>"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
        }