from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # "customer" or "admin". New self-registrations always get "customer",
    # there's no signup path that lets someone set their own role to admin.
    role = db.Column(db.String(20), nullable=False, default="customer")

    # The Messenger Platform ID for this user, used to send messages
    # nullable=True because not all users will have a Messenger PSID
    # unique=True because each Messenger PSID corresponds to exactly one user in our system
    messenger_psid = db.Column(db.String(100), unique=True, nullable=True)

    # do user.orders to see order history.
    # Kept as "customer" here (not "user") since, domain-wise, this
    # relationship represents "the customer who placed this order" 
    orders = db.relationship("Order", back_populates="customer")
    # do user.cart to see the items in their shopping cart
    # use uselist=False to indicate that a user can only have one cart at a time
    cart = db.relationship("Cart", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def set_password(self, plain_password: str):
        # Never store the actual password, only a one-way hash of it.
        self.password_hash = generate_password_hash(plain_password)

    def check_password(self, plain_password: str) -> bool:
        # Hashes entered password the same way and compares  with stored hashed password
        return check_password_hash(self.password_hash, plain_password)

    def __repr__(self):
        return f"<User {self.id} {self.name} ({self.role})>"

    def to_dict(self):
        # Deliberately excludes password_hash, to_dict() is used for
        # dashboard display and any JSON responses; the hash is never
        # sent to a browser
        return {"id": self.id, "name": self.name, "email": self.email, "role": self.role}