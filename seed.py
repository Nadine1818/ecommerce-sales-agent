from app import create_app
from app.extensions import db
from app.models import Category, Product, Customer


def seed():
    app = create_app()

    with app.app_context():
        # Wipe and reseed so the script is safe to re-run during development
        db.drop_all()
        db.create_all()

        electronics = Category(name="Electronics")
        accessories = Category(name="Accessories")
        # adding to the session but not committing yet in case smth fails halfway through
        db.session.add_all([electronics, accessories])
        db.session.flush()
        # flush() sends these INSERTs to the DB and assigns them real ids,
        # without fully committing yet since we need electronics.id and
        # accessories.id to exist before we can reference them below.

        products = [
            Product(
                name="Wireless Mouse",
                description="Ergonomic wireless mouse with 6-month battery life.",
                price=19.99,
                stock_quantity=50,
                # referencing the Category object directly instead of using an id, SQLAlchemy figures out the foreign key id from the relationship automatically.
                category=accessories,
            ),
            Product(
                name="Mechanical Keyboard",
                description="Compact 65% mechanical keyboard, hot-swappable switches.",
                price=79.99,
                stock_quantity=30,
                category=accessories,
            ),
            Product(
                name="27-inch 4K Monitor",
                description="27-inch 4K IPS monitor, 60Hz, USB-C input.",
                price=249.99,
                stock_quantity=15,
                category=electronics,
            ),
            Product(
                name="Noise-Cancelling Headphones",
                description="Over-ear ANC headphones with 30-hour battery.",
                price=129.99,
                stock_quantity=25,
                category=electronics,
            ),
            Product(
                name="USB-C Hub",
                description="7-in-1 USB-C hub: HDMI, SD card, 3x USB-A, PD passthrough.",
                price=34.99,
                stock_quantity=40,
                category=accessories,
            ),
        ]
        db.session.add_all(products)

        customers = [
            Customer(name="Layla Ahmed", email="layla.ahmed@example.com"),
            Customer(name="Omar Khaled", email="omar.khaled@example.com"),
        ]
        db.session.add_all(customers)

        # Commit all the changes to the database
        db.session.commit()

        print(f"Seeded {len(products)} products, 2 categories, {len(customers)} customers.")


if __name__ == "__main__":
    seed()