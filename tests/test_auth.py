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
        data={"name": "Ada Lovelace", "email": "Ada@Example.com", "password": "s3cret123"},
    )

    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"

    with app.app_context():
        user = User.query.filter_by(email="ada@example.com").first()
        assert user is not None
        assert user.name == "Ada Lovelace"
        assert user.role == "customer"
        # password must never be stored in plaintext
        assert user.password_hash != "s3cret123"
        assert user.check_password("s3cret123")

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


@pytest.mark.parametrize(
    "bad_email",
    ["not-an-email", "missing-domain@", "@missing-local.com", "no-at-sign.com", "spaces in@example.com"],
)
def test_register_invalid_email_format_rejected(client, app, bad_email):
    response = client.post(
        "/auth/register",
        data={"name": "A", "email": bad_email, "password": "password1"},
    )

    assert response.status_code == 200
    assert b"Please enter a valid email address." in response.data
    with app.app_context():
        assert User.query.count() == 0


def test_register_password_too_short_rejected(client, app):
    response = client.post(
        "/auth/register",
        data={"name": "A", "email": "short@example.com", "password": "short1"},
    )

    assert response.status_code == 200
    assert b"Password must be at least 8 characters long." in response.data
    with app.app_context():
        assert User.query.count() == 0


def test_register_password_exactly_minimum_length_accepted(client, app):
    response = client.post(
        "/auth/register",
        data={"name": "A", "email": "exact@example.com", "password": "12345678"},
    )

    assert response.status_code == 302
    with app.app_context():
        assert User.query.filter_by(email="exact@example.com").first() is not None


def test_register_duplicate_email_rejected(client, app):
    client.post(
        "/auth/register",
        data={"name": "First", "email": "dup@example.com", "password": "password1"},
    )
    client.get("/auth/logout")

    response = client.post(
        "/auth/register",
        data={"name": "Second", "email": "dup@example.com", "password": "password2"},
    )

    assert response.status_code == 200
    assert b"already exists" in response.data
    with app.app_context():
        assert User.query.count() == 1


def test_register_duplicate_email_case_insensitive(client, app):
    client.post(
        "/auth/register",
        data={"name": "First", "email": "same@example.com", "password": "password1"},
    )
    client.get("/auth/logout")

    response = client.post(
        "/auth/register",
        data={"name": "Second", "email": "SAME@EXAMPLE.COM", "password": "password2"},
    )

    assert b"already exists" in response.data
    with app.app_context():
        assert User.query.count() == 1


def test_register_email_normalized_lowercase_and_stripped(client, app):
    client.post(
        "/auth/register",
        data={"name": "Padded", "email": "  Padded@Example.COM  ", "password": "password1"},
    )

    with app.app_context():
        assert User.query.filter_by(email="padded@example.com").first() is not None


def test_register_cannot_self_assign_admin_role(client, app):
    """The form has no role field, but nothing stops a crafted POST from
    including one — the route must ignore it and always create customers."""
    response = client.post(
        "/auth/register",
        data={"name": "Wannabe Admin", "email": "wannabe@example.com", "password": "password1", "role": "admin"},
    )

    assert response.status_code == 302
    with app.app_context():
        user = User.query.filter_by(email="wannabe@example.com").first()
        assert user.role == "customer"


def test_register_password_with_leading_trailing_spaces_preserved(client, app):
    """Unlike name/email, password is intentionally NOT stripped — a
    password of " password1 " and "password1" must be treated as different secrets."""
    client.post(
        "/auth/register",
        data={"name": "A", "email": "spacey@example.com", "password": " password1 "},
    )

    with app.app_context():
        user = User.query.filter_by(email="spacey@example.com").first()
        assert user.check_password(" password1 ") is True
        assert user.check_password("password1") is False


def test_register_xss_payload_in_name_is_escaped_on_render(client):
    payload = "<script>alert(1)</script>"
    client.post(
        "/auth/register",
        data={"name": payload, "email": "xss@example.com", "password": "password1"},
    )

    chat_response = client.get("/chat/")
    assert payload.encode() not in chat_response.data
    assert b"&lt;script&gt;" in chat_response.data


# ------------------------------------------------------------------ login

def test_login_page_renders_form(client):
    response = client.get("/auth/login")
    assert response.status_code == 200
    assert b"<form" in response.data


def _register(client, name="Bob", email="bob@example.com", password="hunter22"):
    client.post("/auth/register", data={"name": name, "email": email, "password": password})
    client.get("/auth/logout")


def test_login_success(client):
    _register(client)

    response = client.post("/auth/login", data={"email": "bob@example.com", "password": "hunter22"})

    assert response.status_code == 302
    assert response.headers["Location"] == "/chat/"
    assert client.get("/chat/").status_code == 200


def test_login_wrong_password(client):
    _register(client)

    response = client.post("/auth/login", data={"email": "bob@example.com", "password": "wrong"})

    assert response.status_code == 200
    assert b"Invalid email or password." in response.data
    # must not be logged in — /chat/ itself is open to guests now, so
    # check a route that's still login_required instead
    assert client.get("/cart/").status_code == 302


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

    response = client.post("/auth/login", data={"email": "CASE@Example.com", "password": "hunter22"})

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
    client.post("/auth/login", data={"email": "bob@example.com", "password": "hunter22"})
    assert client.get("/chat/").status_code == 200

    response = client.get("/auth/logout")

    assert response.status_code == 302
    assert response.headers["Location"] == "/auth/login"
    # /chat/ is open to guests now, so it's not a useful check here —
    # /cart/ is still login_required, so it's what actually proves the
    # session was cleared.
    assert client.get("/cart/").status_code == 302
    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_logout_without_active_session_does_not_crash(client):
    response = client.get("/auth/logout")
    assert response.status_code == 302


# --------------------------------------------------------- login_required

def test_chat_index_accessible_to_guests(client):
    """chat.index has no login_required: anyone can view and use the
    chat. Only add_to_cart/create_order are actually gated, and that
    gating lives in sales_node's tool binding (see test_tools.py /
    app/agent/nodes.py), not in this route."""
    response = client.get("/chat/")
    assert response.status_code == 200
    # guest header, not the logged-in user block
    assert b"Browsing as guest" in response.data

    # sending a message isn't exercised here since it invokes the real
    # LLM (no mocking infrastructure exists in this suite for that) —
    # covered instead by unit tests on the tools/nodes themselves.


def test_admin_can_also_view_chat(client, app):
    """chat is open to everyone now, including an admin account — there's
    no role restriction on the route itself anymore. Login itself still
    redirects an admin to /dashboard/ (unrelated to today's changes,
    see auth/routes.py's role check) — this test is about chat.index
    being reachable afterward, not about where login lands."""
    with app.app_context():
        admin = User(name="Root", email="admin@example.com", role="admin")
        admin.set_password("adminpw")
        db.session.add(admin)
        db.session.commit()

    login_response = client.post(
        "/auth/login", data={"email": "admin@example.com", "password": "adminpw"}
    )
    assert login_response.status_code == 302
    assert login_response.headers["Location"] == "/dashboard/"

    chat_response = client.get("/chat/")
    assert chat_response.status_code == 200