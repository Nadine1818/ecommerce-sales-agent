# A single decorator, reused for every protected route in the app
from functools import wraps

from flask import flash, redirect, session, url_for

# which role should this route require
def login_required(role: str = None):
    # which route function is being protected by this decorator
    def decorator(view_func):
        @wraps(view_func)
        # is this specific visior allowed access to this route 
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.")
                return redirect(url_for("auth.login"))
            # if role is specified but the logged-in user doesn't have that role, redirect to login page
            if role is not None and session.get("role") != role:
                flash("You don't have access to that page.")
                return redirect(url_for("auth.login"))

            return view_func(*args, **kwargs)

        return wrapped

    return decorator