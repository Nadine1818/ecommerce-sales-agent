"""Register/login/logout routes. Registration always creates role="customer"
accounts , there's no signup path that lets someone become an admin,
the one admin account is seeded directly in seed.py"""

from flask import flash, redirect, render_template, request, session, url_for

from app.auth import auth_bp
from app.extensions import db
from app.models import User

# GET to /register shows the registration form, POST to /register processes it
@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.")
            return render_template("register.html")

        user = User(name=name, email=email, role="customer")
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Log the new user in immediately, same as a successful login would.
        session["user_id"] = user.id
        session["role"] = user.role
        session["name"] = user.name
        # redirect to the chat page
        return redirect(url_for("chat.index"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        # check_password is only called if a user was actually found,
        # calling it on None would crash, a wrong email
        # and a wrong password produce the same generic error message
        if user is None or not user.check_password(password):
            flash("Invalid email or password.")
            return render_template("login.html")

        session["user_id"] = user.id
        session["role"] = user.role
        session["name"] = user.name

        if user.role == "admin":
            return redirect(url_for("dashboard.index"))
 
        return redirect(url_for("chat.index"))

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.")
    return redirect(url_for("auth.login"))