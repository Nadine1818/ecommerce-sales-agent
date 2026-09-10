from app.extensions import db


class Customer(db.Model):
    __tablename__ = "customers"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True)

    # customer.orders will return a list of Order objects that belong to this customer.
    orders = db.relationship("Order", back_populates="customer")

    def __repr__(self):
        return f"<Customer {self.id} {self.name}>"

    def to_dict(self):
        return {"id": self.id, "name": self.name, "email": self.email}