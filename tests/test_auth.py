"""Tests for register/login/logout (app/auth/routes.py) and the
login_required decorator, including edge cases: duplicate/whitespace/case
handling on email, role self-assignment, SQL-injection-shaped input, and
XSS-shaped input in stored fields."""

import pytest

from app.extensions import db
from app.models import User


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------- register

def test_register_page_renders_form(client):
    response = client.get("/auth/register")
    assert response.status_code == 200
    assert b"<form" in response.data


def test_register_success_creates_user_and_logs_in(client, app):
    response = client.post(
        "/auth/register",
        data={"name": "Ada Lovelace", "email": "Ada@Example.com", "password": "s3cret"},
    )

    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"

    with app.app_context():
        user = User.query.filter_by(email="ada@example.com").first()
        assert user is not None
        assert user.name == "Ada Lovelace"
        assert user.role == "customer"
        # password must never be stored in plaintext
        assert user.password_hash != "s3cret"
        assert user.check_password("s3cret")

    # the register call above logs the user straight in
    chat_response = client.get("/chat/")
    assert chat_response.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "email": "a@example.com", "password": "pw"},
        {"name": "A", "email": "", "password": "pw"},
        {"name": "A", "email": "a@example.com", "password": ""},
        {"name": "   ", "email": "a@example.com", "password": "pw"},  # whitespace-only
        {"name": "A", "email": "   ", "password": "pw"},
    ],
)
def test_register_missing_or_blank_fields_rejected(client, app, payload):
    response = client.post("/auth/register", data=payload)

    assert response.status_code == 200  # re-renders the form, no redirect
    assert b"All fields are required." in response.data
    with app.app_context():
        assert User.query.count() == 0


def test_register_duplicate_email_rejected(client, app):
    client.post(
        "/auth/register",
        data={"name": "First", "email": "dup@example.com", "password": "pw1"},
    )
    client.get("/auth/logout")

    response = client.post(
        "/auth/register",
        data={"name": "Second", "email": "dup@example.com", "password": "pw2"},
    )

    assert response.status_code == 200
    assert b"already exists" in response.data
    with app.app_context():
        assert User.query.count() == 1


def test_register_duplicate_email_case_insensitive(client, app):
    client.post(
        "/auth/register",
        data={"name": "First", "email": "same@example.com", "password": "pw1"},
    )
    client.get("/auth/logout")

    response = client.post(
        "/auth/register",
        data={"name": "Second", "email": "SAME@EXAMPLE.COM", "password": "pw2"},
    )

    assert b"already exists" in response.data
    with app.app_context():
        assert User.query.count() == 1


def test_register_email_normalized_lowercase_and_stripped(client, app):
    client.post(
        "/auth/register",
        data={"name": "Padded", "email": "  Padded@Example.COM  ", "password": "pw"},
    )

    with app.app_context():
        assert User.query.filter_by(email="padded@example.com").first() is not None


def test_register_cannot_self_assign_admin_role(client, app):
    """The form has no role field, but nothing stops a crafted POST from
    including one — the route must ignore it and always create customers."""
    response = client.post(
        "/auth/register",
        data={"name": "Wannabe Admin", "email": "wannabe@example.com", "password": "pw", "role": "admin"},
    )

    assert response.status_code == 302
    with app.app_context():
        user = User.query.filter_by(email="wannabe@example.com").first()
        assert user.role == "customer"


def test_register_password_with_leading_trailing_spaces_preserved(client, app):
    """Unlike name/email, password is intentionally NOT stripped — a
    password of " pw " and "pw" must be treated as different secrets."""
    client.post(
        "/auth/register",
        data={"name": "A", "email": "spacey@example.com", "password": " pw "},
    )

    with app.app_context():
        user = User.query.filter_by(email="spacey@example.com").first()
        assert user.check_password(" pw ") is True
        assert user.check_password("pw") is False


def test_register_xss_payload_in_name_is_escaped_on_render(client):
    payload = "<script>alert(1)</script>"
    client.post(
        "/auth/register",
        data={"name": payload, "email": "xss@example.com", "password": "pw"},
    )

    chat_response = client.get("/chat/")
    assert payload.encode() not in chat_response.data
    assert b"&lt;script&gt;" in chat_response.data


# ------------------------------------------------------------------ login

def test_login_page_renders_form(client):
    response = client.get("/auth/login")
    assert response.status_code == 200
    assert b"<form" in response.data


def _register(client, name="Bob", email="bob@example.com", password="hunter2"):
    client.post("/auth/register", data={"name": name, "email": email, "password": password})
    client.get("/auth/logout")


def test_login_success(client):
    _register(client)

    response = client.post("/auth/login", data={"email": "bob@example.com", "password": "hunter2"})

    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"
    assert client.get("/chat/").status_code == 200


def test_login_wrong_password(client):
    _register(client)

    response = client.post("/auth/login", data={"email": "bob@example.com", "password": "wrong"})

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data
    # must not be logged in
    assert client.get("/chat/").status_code == 302


def test_login_nonexistent_email(client):
    response = client.post("/auth/login", data={"email": "nobody@example.com", "password": "pw"})

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_login_nonexistent_and_wrong_password_give_identical_message(client):
    """Route comment says this is deliberate (no user enumeration).
    Verify both failure paths really do produce the same flash text."""
    _register(client)

    wrong_password = client.post("/auth/login", data={"email": "bob@example.com", "password": "nope"})
    no_such_user = client.post("/auth/login", data={"email": "ghost@example.com", "password": "nope"})

    def flash_text(resp):
        start = resp.data.index(b'class="flash"')
        return resp.data[start:start + 200]

    assert b"Invalid email or password." in wrong_password.data
    assert b"Invalid email or password." in no_such_user.data


def test_login_email_case_insensitive(client):
    _register(client, email="case@example.com")

    response = client.post("/auth/login", data={"email": "CASE@Example.com", "password": "hunter2"})

    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"


def test_login_password_is_case_sensitive(client):
    _register(client)

    response = client.post("/auth/login", data={"email": "bob@example.com", "password": "HUNTER2"})

    assert b"Invalid email or password." in response.data


def test_login_empty_credentials(client):
    response = client.post("/auth/login", data={"email": "", "password": ""})

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_login_sql_injection_shaped_email_is_handled_safely(client):
    """The ORM parameterizes the query, so a classic injection string is
    just an email that doesn't match anything — no crash, no bypass."""
    response = client.post(
        "/auth/login",
        data={"email": "' OR '1'='1", "password": "' OR '1'='1"},
    )

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


def test_login_does_not_crash_on_missing_form_fields(client):
    """POST with no body at all, not even empty strings."""
    response = client.post("/auth/login", data={})

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data


# ----------------------------------------------------------------- logout

def test_logout_clears_session(client):
    _register(client)
    client.post("/auth/login", data={"email": "bob@example.com", "password": "hunter2"})
    assert client.get("/chat/").status_code == 200

    response = client.get("/auth/logout")

    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"
    # session is gone, protected route bounces back to login
    assert client.get("/chat/").status_code == 302
    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_logout_without_active_session_does_not_crash(client):
    response = client.get("/auth/logout")
    assert response.status_code == 302


# --------------------------------------------------------- login_required

def test_chat_index_requires_login(client):
    response = client.get("/chat/")
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_chat_send_requires_login(client):
    response = client.post("/chat/send", json={"message": "hi"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"


def test_admin_role_is_locked_out_of_customer_chat(client, app):
    """chat.index is decorated with login_required(role='customer'), and
    there's no admin dashboard route yet. An admin who logs in is
    redirected to /chat/ by the login view, then immediately bounced
    back to /auth/login by the decorator on that same page."""
    with app.app_context():
        admin = User(name="Root", email="admin@example.com", role="admin")
        admin.set_password("adminpw")
        db.session.add(admin)
        db.session.commit()

    login_response = client.post(
        "/auth/login", data={"email": "admin@example.com", "password": "adminpw"}
    )
    assert login_response.status_code == 302
    assert login_response.headers["Location"] == "/chat/"

    chat_response = client.get("/chat/")
    assert chat_response.status_code == 302
    assert chat_response.headers["Location"] == "/auth/login"
