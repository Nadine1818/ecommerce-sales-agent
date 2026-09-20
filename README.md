# QuickShelf — AI Sales & Customer Service Agent

An AI-powered sales and customer service agent for **QuickShelf**, a small e-commerce store selling electronics and accessories. The agent understands natural-language customer messages, retrieves accurate information from a knowledge base using **RAG**, and can take a real business action — placing an order — through **LangGraph**-orchestrated tool calls. An admin dashboard built with **Flask** lets a store administrator manage products, customer accounts, and the knowledge base, and view order history, all without touching any code.

Built as a technical assessment. This README documents the business domain, the architecture, how the agent and RAG pipeline work, the database design, the available tools, how to run everything locally, and known limitations.

---

## Table of Contents

1. [Business Domain](#1-business-domain)
2. [Architecture Overview](#2-architecture-overview)
3. [The LangGraph Agent](#3-the-langgraph-agent)
4. [RAG (Retrieval-Augmented Generation)](#4-rag-retrieval-augmented-generation)
5. [Database](#5-database)
6. [Tools / Business Actions](#6-tools--business-actions)
7. [Admin Dashboard](#7-admin-dashboard)
8. [Bonus: Facebook Messenger Integration](#8-bonus-facebook-messenger-integration)
9. [Project Structure](#9-project-structure)
10. [Getting Started (Run Locally)](#10-getting-started-run-locally)
11. [Environment Variables](#11-environment-variables)
12. [Running the Tests](#12-running-the-tests)
13. [Example Conversations](#13-example-conversations)
14. [Limitations & Assumptions](#14-limitations--assumptions)
15. [Tech Stack Summary](#15-tech-stack-summary)

---

## 1. Business Domain

**QuickShelf** is a small e-commerce store selling electronics and accessories (mice, keyboards, monitors, headphones, hubs, etc.). It was chosen because it maps cleanly onto every required capability in the assessment:

- A **product catalog** with categories, prices, descriptions, and live stock — a natural fit for sales conversations and recommendations *(surfaced via similarity search over the catalog, not a dedicated recommendation algorithm)*.
- **FAQs and store policies** (shipping, returns, warranties, payment methods) — a natural fit for customer-service conversations.
- A clear, unambiguous **business action** — placing an order — that actually changes state in the database (decrements stock, creates order records) rather than just being described by the LLM.

The agent serves two kinds of customers: **guests**, who can freely browse and ask questions, and **logged-in customers**, who can additionally build a cart and place real orders.

---

## 2. Architecture Overview

```
                         ┌─────────────────────────┐
                         │   Customer (Browser)     │
                         │  or Facebook Messenger   │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Flask Application       │
                         │ ───────────────────────── │
                         │  Blueprints:               │
                         │  auth · chat · cart ·      │
                         │  orders · dashboard ·      │
                         │  messenger                 │
                         └────────────┬─────────────┘
                                      │  invokes on every chat turn
                         ┌────────────▼─────────────┐
                         │   Compiled LangGraph      │
                         │        Agent               │
                         │  classify_intent           │
                         │        │                    │
                         │  ┌─────┴──────┐             │
                         │  │            │             │
                         │ sales_node  customer_       │
                         │            service_node     │
                         │  │            │             │
                         │  └─────┬──────┘             │
                         │   format_response            │
                         └──┬──────────────┬───────────┘
                            │              │
                 ┌──────────▼───┐   ┌──────▼───────────┐
                 │  RAG (Chroma  │   │  Business tools    │
                 │  + BGE        │   │  (SQLAlchemy /      │
                 │  embeddings)  │   │  SQLite)            │
                 └───────────────┘   └────────────────────┘
                            │              │
                         ┌──▼──────────────▼───┐
                         │   SQLite Database     │
                         │  (via SQLAlchemy ORM,  │
                         │   Flask-Migrate)       │
                         └───────────────────────┘
```

The Flask app is organized as **blueprints**, one per concern:

| Blueprint    | URL prefix     | Responsibility                                                             |
|--------------|-----------------|------------------------------------------------------------------------------|
| `auth`       | `/auth`         | Registration, login, logout, session management                          |
| `chat`       | `/chat`         | The customer-facing chat UI and the endpoint that invokes the agent graph |
| `cart`       | `/cart`         | Viewing/editing the persistent cart, checkout                            |
| `orders`     | `/orders`       | A logged-in customer's own order history                                 |
| `dashboard`  | `/dashboard`    | Admin CRUD for products and customers, order history (view-only), and full RAG data management |
| `messenger`  | `/messenger`    | The Facebook Messenger webhook (bonus feature)                           |

The same **compiled LangGraph agent** (`app.agent.compiled_graph`) is invoked from both `chat/routes.py` (web chat) and `messenger/routes.py` (Facebook), so there is exactly one agent implementation behind both surfaces.

---

## 3. The LangGraph Agent

The agent is a `StateGraph` (`app/agent/graph.py`) built around a single shared `AgentState` (`app/agent/state.py`):

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full LangChain conversation history
    customer_id: Optional[int]                 # None for a guest, the user's id otherwise
    intent: Optional[str]                       # "sales" | "customer_service"
    retrieved_context: Optional[list]           # last RAG result, for inspection/debugging
    tool_result: Optional[dict]                 # result of create_order / request_login
    response: Optional[str]                     # final text sent back to the customer
```

**Graph flow:**

```
User Message
     │
     ▼
classify_intent  ──────────────►  (conditional edge on state["intent"])
     │
     ├── "sales" ─────────────►  sales_node
     └── "customer_service" ──►  customer_service_node
                                        │
                    both branches converge on
                                        ▼
                                format_response
                                        │
                                        ▼
                                       END
```

**Node by node:**

- **`classify_intent`** — Sends the *full* conversation history (not just the latest message) to the LLM with a narrow instruction to answer with exactly one word: `sales` or `customer_service`. Using the full history matters because a short follow-up like `"2"` or `"the second one"` is meaningless in isolation — it only makes sense in light of what was asked before it. If the LLM call fails after retries, or returns anything other than one of the two expected words, the node **defaults to `customer_service`** (the safer, non-purchasing path) and logs a warning rather than crashing the request.

- **`sales_node`** — Binds a different set of tools depending on whether `customer_id` is set:
  - **Guest** (`customer_id is None`): `retrieve_product_info`, `check_product_availability`, `request_login`. There is no `add_to_cart`/`create_order` tool bound at all, so a guest is *structurally* unable to place an order — not merely told not to. The system prompt instructs the LLM to call `request_login` the moment the guest expresses any intent to buy, regardless of what the guest claims about being logged in (identity is decided by which tools are bound, never by what the user says in the chat).
  - **Logged-in customer**: `retrieve_product_info`, `check_product_availability`, `add_to_cart`, `create_order`. The `customer_id` is injected server-side into every tool call — the LLM cannot be talked into acting on a different account, even if the customer names another account by id or by name.

- **`customer_service_node`** — Binds a single tool, `retrieve_support_info` (RAG over FAQs and policies), and is instructed to answer only from what that tool returns, saying so honestly if the retrieved information doesn't fully cover the question, rather than inventing an answer.

- **`format_response`** — Pulls the agent's final message out of the conversation history into `state["response"]`, giving the Flask routes one predictable field to read regardless of which branch produced it.

**Multi-step tool-calling loop.** Both `sales_node` and `customer_service_node` run through a shared `_run_agent_loop` (in `app/agent/nodes.py`) rather than a fixed "one LLM call → maybe one tool call → one final call" shape. It repeatedly calls the LLM and executes whatever tools it requests, feeding the results back in, until the LLM responds with no further tool calls or a safety cap of **4 iterations** is hit. This lets the agent chain tool calls that genuinely depend on each other — e.g. calling `retrieve_product_info` to resolve a product name to its `product_id`, then using that id in a follow-up `add_to_cart` or `create_order` call — which a single fixed round-trip can't express, since the customer refers to products by name and the id genuinely isn't known until the first tool's result comes back. (`check_product_availability` is a separate, standalone tool for re-checking stock on a product already discussed earlier in the conversation — `retrieve_product_info` already returns live stock for anything it looks up, so the two aren't normally chained together in the same turn.)

**Resilience.** Groq's `openai/gpt-oss-120b` occasionally returns an empty or unparseable completion (a known provider-side flakiness). Every raw LLM call goes through `_invoke_llm_with_retry`, which retries up to 3 times with a short backoff. Only the bare LLM call is retried — never a tool with a side effect — so a retry can never double-place an order or double-add a cart line. If the whole agent invocation still fails, the chat/Messenger routes catch the exception and return a clean "temporarily unavailable" message instead of a stack trace.

---

## 4. RAG (Retrieval-Augmented Generation)

**Stack:** [ChromaDB](https://www.trychroma.com/) (persisted to `data/chroma/`) as the vector store, and `BAAI/bge-small-en-v1.5` (via `langchain-huggingface` / `sentence-transformers`) as the embedding model — a small, free, local model with no API key required.

**Content in the knowledge base:**

| Type      | Source                                                                 | Example                                    |
|-----------|-------------------------------------------------------------------------|---------------------------------------------|
| `product` | The live `Product` table (name, category, price, description)         | "Wireless Mouse", "27-inch 4K Monitor"     |
| `faq`     | `app/rag/knowledge_data.py` (seed) + anything added via the dashboard  | "Do you ship internationally?"             |
| `policy`  | `app/rag/knowledge_data.py` (seed) + anything added via the dashboard  | Shipping Policy, Return Policy, Delivery Information |

Every item is embedded as a single descriptive string (e.g. a product becomes `"Product: {name}. Category: {category}. Price: ${price}. Description: {description}"`), tagged in ChromaDB metadata with a `type` field (`product` / `faq` / `policy`) plus the raw source fields. That `type` tag is what lets `retrieve()` filter a search to only one or several content types.

**Retrieval (`app/rag/retrieve.py`)** — `retrieve(query, top_k, item_type)` embeds the query with the same model used at ingestion time (so both live in the same vector space), then runs a similarity search against ChromaDB, optionally filtered by `type`. Two tools wrap this for the agent:
- `retrieve_product_info` → `item_type="product"`, `top_k=10` (higher than the current 5-product catalog so a broad query like "what do you have?" can surface everything, not just the 3 nearest matches — at a much larger catalog this would be tuned back down, or split into a separate browse path rather than raised indefinitely)
- `retrieve_support_info` → `item_type=["faq", "policy"]`, `top_k=3`

`get_all_items()` does a plain, non-similarity fetch of everything of a given type — used by the dashboard's RAG data page to list every entry, not just ones matching a query.

**RAG management (`app/rag/ingest.py`)** — this is what satisfies the "add / update / delete, and use the updated data immediately" requirement:
- `add_or_update_item(item_id, item_type, data)` — computes the embedding and `upsert`s it into ChromaDB. Because it's an upsert keyed on `item_id`, the *same* function handles both "add" and "update" — if the id already exists its vector and text are simply replaced.
- `delete_item(item_id)` — removes an entry by id, returns whether it existed.
- `ingest_all(items, item_type)` — bulk-ingests a list (used for the initial seed data and for products).

Because ChromaDB is a normal Python object the app talks to directly (not swapped out per-request), any add/edit/delete from the dashboard is reflected in the very next retrieval — there's no separate "republish" step.

**How new content gets into the RAG store:**
- **Products** added, edited, or deleted through the **admin dashboard** (`/dashboard/products`) sync into RAG immediately — the same `add_or_update_item`/`delete_item` calls the RAG management pages use — so a product created there is searchable right away, no script needed. For products that enter the database another way (e.g. `seed.py`, or a bulk import outside the dashboard), running `python ingest_knowledge_base.py` picks them up: it ingests every product currently in the SQL `Product` table and also *reconciles deletions* — it computes the current set of product ids and removes any stale product vectors left over from products no longer in the database (covering products removed by anything other than the dashboard's own delete, which already cleans up its own RAG entry immediately).
- **FAQs and policies** start from the hardcoded seed list in `app/rag/knowledge_data.py`, but from that point on are fully managed live through the **admin dashboard** (add / edit / delete), which calls `ingest.py` directly — no re-running of any script needed.

---

## 5. Database

**SQLite** (`data/app.db`) via **Flask-SQLAlchemy** as the ORM, with **Flask-Migrate / Alembic** for schema migrations (a baseline migration is checked into `migrations/versions/`). `db.create_all()` is still called at app startup as a safety net (mainly so the test suite's in-memory database always has tables even without running a migration), but the source of truth for schema changes going forward is `flask db migrate` / `flask db upgrade`.

**Entities:**

| Model        | Table          | Key fields                                                              | Relationships                                    |
|--------------|-----------------|---------------------------------------------------------------------------|----------------------------------------------------|
| `Category`   | `categories`   | `id`, `name` (unique)                                                    | has many `Product`                                 |
| `Product`    | `products`     | `id`, `name`, `description`, `price`, `stock_quantity`, `category_id`   | belongs to `Category`                              |
| `User`       | `users`        | `id`, `name`, `email` (unique), `password_hash`, `role`, `messenger_psid` (unique, nullable) | has many `Order`; has one `Cart`     |
| `Cart`       | `carts`        | `id`, `user_id` (unique — one cart per user), `created_at`              | belongs to `User`; has many `CartItem`             |
| `CartItem`   | `cart_items`   | `id`, `cart_id`, `product_id`, `quantity`                                | belongs to `Cart`, references `Product`            |
| `Order`      | `orders`       | `id`, `customer_id`, `status`, `created_at`, `total_price`               | belongs to `User`; has many `OrderItem`            |
| `OrderItem`  | `order_items`  | `id`, `order_id`, `product_id`, `quantity`, `price_at_order`             | belongs to `Order`, references `Product`           |

A few deliberate design choices worth calling out:

- **`price_at_order` is a snapshot**, not a live join to `Product.price`. If a product's price changes later, past orders still reflect what the customer actually paid at the time.
- **Stock is decremented atomically at order time** with a single conditional `UPDATE ... WHERE stock_quantity >= :qty`, not a separate read-then-write. This prevents two simultaneous orders from both reading "5 in stock" and both succeeding when only 5 units actually exist — the update fails (0 rows affected) instead of overselling.
- **`Cart` enforces one cart per user** at the database level via a unique constraint on `user_id`, created lazily the first time a customer adds something.
- **`User.role`** is either `"customer"` or `"admin"`. There is no self-registration path to `admin` — the one admin account is seeded directly (see `seed.py`).

---

## 6. Tools / Business Actions

These are the functions the LLM can choose to call (via LangChain's `@tool` decorator, bound per-node as described in [Section 3](#3-the-langgraph-agent)):

| Tool                          | Bound in                                | What it does                                                                                          | Real side effect? |
|-------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------------|:---:|
| `retrieve_product_info`       | sales (guest & logged-in)                 | RAG search over products; includes live stock in every result                                        | No (read-only) |
| `check_product_availability`  | sales (guest & logged-in)                 | Re-checks stock for a known `product_id` without a fresh search                                       | No (read-only) |
| `retrieve_support_info`       | customer service                          | RAG search over FAQs + policies                                                                       | No (read-only) |
| `request_login`               | sales (guest only)                        | Signals the frontend to prompt a login — the guest-safe stand-in for "I want to buy something"        | No |
| `add_to_cart`                 | sales (logged-in only)                    | Adds/increments a line in the customer's persistent cart, after a stock check                         | **Yes** — writes to `carts`/`cart_items` |
| `create_order`                | sales (logged-in only)                    | **The required business action.** Places a real order: atomically decrements stock per item, creates `Order`/`OrderItem` rows, computes the total, and syncs the customer's cart to remove/decrement whatever was just ordered | **Yes** — writes to `orders`, `order_items`, `products.stock_quantity`, and `cart_items` |

`create_order` is also called directly by the **`/cart/checkout`** web route, so a checkout-page order and a chat-agent order both go through the exact same stock/price logic — there is only one implementation of "place an order" in the whole app.

If the LLM ever requests a tool that isn't actually bound for the current turn (e.g. it remembers calling `request_login` earlier and tries to call it again after the customer has since logged in), the agent feeds back a plain error message the LLM can read and self-correct from, instead of crashing the request.

---

## 7. Admin Dashboard

Built with Flask + Jinja templates, protected end-to-end by a `login_required(role="admin")` decorator — a logged-in customer cannot reach any dashboard route, nor can an anonymous visitor.

The dashboard provides full CRUD for products, customer accounts, and RAG data, with orders kept read-only:

- **Products — full CRUD** (`/dashboard/products`): add, edit, and delete products. Add/edit immediately embeds the product into the RAG knowledge base too (same `add_or_update_item` call the RAG management pages use), so a newly added product is searchable by the chat agent right away, with no separate `ingest_knowledge_base.py` run needed. Deleting a product that appears in existing order history is blocked (order pages already fall back to "(deleted product)"/"Product #<id>" for a genuinely missing product, but the dashboard doesn't let that happen by default) — deleting a product only in someone's cart is allowed and clears the matching cart line, and its RAG entry is removed at the same time.
- **Customers — full CRUD** (`/dashboard/customers`): add, edit, and delete customer accounts. This is deliberately scoped to `role="customer"` only — there is still no path, including this one, that creates or edits an *admin* account from the UI (the one admin account remains seeded directly via `seed.py`), preserving the original "no self-service path to admin" design. Deleting a customer with existing order history is blocked for the same reason as products; deleting one with no orders cascades their cart automatically (`User.cart`'s existing `cascade="all, delete-orphan"` handles this).
- **Orders — view-only** (`/dashboard/orders`): orders remain read-only by design — an order is a historical record of what was actually purchased, and editing it after the fact isn't something a real store would want either.
- **RAG data — full CRUD**:
  - `/dashboard/rag` — lists every FAQ and policy currently in the knowledge base
  - `/dashboard/rag/add` — add a new FAQ or policy (immediately embedded and searchable)
  - `/dashboard/rag/edit/<item_id>` — edit an existing entry in place (re-embeds on save)
  - `/dashboard/rag/delete/<item_id>` — delete an entry (POST-only, guarded by a confirmation modal in the UI, so a page refresh or stray link click can never trigger a deletion)

Every delete action across the dashboard (products, customers, and RAG entries) goes through the same POST-only route + confirmation-modal pattern, so nothing is ever deletable via a GET request or an accidental click. A `threading.Lock` guards the "compute the next numeric id, then write it" sequence in `rag_add`, so two admins adding an FAQ at the same moment can't be handed the same id.

---

## 8. Bonus: Facebook Messenger Integration

Implemented as an additional, optional bonus feature — the same compiled LangGraph agent used by the web chat, reused unchanged.

```
Customer → Facebook Messenger → Meta Webhook → Flask (/messenger/webhook)
    → LangGraph Agent → RAG / Tools → Response → Facebook Messenger → Customer
```

- `GET /messenger/webhook` handles Meta's one-time verification handshake (echoes back `hub.challenge` if `hub.verify_token` matches `VERIFY_TOKEN`).
- `POST /messenger/webhook` receives every message event, resolves it to a `User`, runs the same agent graph, and sends the reply back via the Graph API using `PAGE_ACCESS_TOKEN`.
- **Account linking** — Messenger identifies people by an opaque per-page PSID, not an email, so a PSID has to be linked to a real `User.id` before the agent can look up that customer's cart/orders. A logged-in customer sees a **"Chat on Messenger"** link on the web chat page (`https://m.me/<page>?ref=<user_id>`). Two mechanisms link the PSID, coexisting in the webhook code:
  1. **Primary (invisible):** Meta's own `?ref=` referral parameter, read straight off the first incoming message event — reliable for a genuinely first-time conversation with the Page.
  2. **Fallback:** if the customer types a message matching `LINK-<user_id>` (the deep link's pre-filled text), the webhook links from that instead. This covers accounts with pre-existing conversation history with the Page, where Meta's referral delivery is a known, documented platform reliability issue.
- Conversation history for Messenger users is kept in an in-process dict keyed by `customer_id` (see [Limitations](#14-limitations--assumptions) — this is intentionally simple for a bonus feature, not the persistent session store the web chat uses).

---

## 9. Project Structure

```
ecommerce-sales-agent/
├── app/
│   ├── agent/            # LangGraph graph, nodes, state, tools
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   ├── state.py
│   │   └── tools.py
│   ├── auth/              # register / login / logout, login_required decorator
│   ├── cart/               # cart view, update, remove, checkout
│   ├── chat/               # chat UI + /send endpoint that invokes the agent
│   ├── dashboard/          # admin CRUD for products/customers + RAG, orders view-only
│   ├── messenger/          # Facebook Messenger webhook (bonus)
│   ├── models/              # SQLAlchemy models (Category, Product, User, Cart, CartItem, Order, OrderItem)
│   ├── rag/                 # ChromaDB store, ingest, retrieve, seed knowledge_data.py
│   ├── static/               # CSS/JS for the templates
│   └── __init__.py          # app factory (create_app)
├── migrations/              # Flask-Migrate / Alembic migrations
├── tests/                   # pytest suite (unit + integration)
│   └── manual/               # small standalone smoke-test scripts
├── ingest_knowledge_base.py  # (re)ingests products/FAQs/policies into ChromaDB
├── seed.py                   # resets the DB and seeds demo categories/products/users
├── run.py                    # entry point (flask dev server)
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## 10. Getting Started (Run Locally)

**Prerequisites:** Python 3.11+, and a free [Groq](https://console.groq.com/) API key (the LLM provider used here).

```bash
# 1. Clone and enter the project
git clone https://github.com/Nadine1818/ecommerce-sales-agent.git
cd ecommerce-sales-agent

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then open .env and set GROQ_API_KEY (see Section 11)

# 5. Create and seed the database (categories, products, a customer, and an admin)
python seed.py

# 6. Embed the seeded products, FAQs, and policies into ChromaDB
python ingest_knowledge_base.py

# 7. Run the app
python run.py
```

The app starts at **http://localhost:5000**. On the first request, `python run.py` warms up the BGE embedding model up front (a one-time cost) so the first real chat message isn't the one paying for it.

- Chat as a **guest**: `http://localhost:5000/chat`
- Log in as the **seeded customer**: `laylaAhmed@gmail.com` / `12345678` (or `omarKhaled@gmail.com` / `12345678`)
- Log in as the **seeded admin**: `admin@gmail.com` / `12345678` → redirects straight to the dashboard

> **Note on `seed.py` vs. migrations:** `seed.py` calls `db.drop_all()` / `db.create_all()` directly from the SQLAlchemy models — it's the fast path for local development and is safe to re-run at any time to reset to a clean demo state. The migration-tracked path (`flask db upgrade`, with `FLASK_APP=run.py` set) is also available and is what a real deployment would use to evolve the schema over time without dropping data. If you re-run `seed.py`, re-run `python ingest_knowledge_base.py` afterwards too, since the products in ChromaDB need to stay in sync with whatever product ids currently exist in `data/app.db`.

---

## 11. Environment Variables

Defined in `.env` (see `.env.example`):

| Variable                | Required?                     | Purpose                                                                 |
|--------------------------|-------------------------------|---------------------------------------------------------------------------|
| `SECRET_KEY`             | Recommended (has a dev default) | Flask session signing key                                              |
| `GROQ_API_KEY`           | **Yes**                        | Authenticates calls to the Groq LLM (`openai/gpt-oss-120b`) that powers the agent |
| `PAGE_ACCESS_TOKEN`      | Only for the Messenger bonus  | Authenticates outgoing replies to the Facebook Graph API                  |
| `VERIFY_TOKEN`           | Only for the Messenger bonus  | Shared secret Meta uses during the webhook verification handshake         |
| `MESSENGER_PAGE_USERNAME`| Only for the Messenger bonus  | Used to build the `m.me/<username>?ref=<user_id>` deep link shown to logged-in customers |

The core web chat, RAG, orders, and dashboard all work with only `GROQ_API_KEY` set — the Messenger variables are only needed if you're wiring up the bonus integration.

---

## 12. Running the Tests

```bash
pytest
```

The suite (243 test functions across `tests/test_agent.py`, `test_auth.py`, `test_cart.py`, `test_chat.py`, `test_dashboard.py`, `test_messenger.py`, `test_models.py`, `test_orders.py`, `test_rag.py`, and `test_tools.py`) runs entirely against a fresh, isolated **in-memory SQLite database** per test (never `data/app.db`), so it's always safe to run and never touches real data.

- **No `GROQ_API_KEY` is required to run the tests.** Agent/LLM-dependent tests substitute a `FakeLLM` for `get_llm()`, so classification and tool-calling logic are tested deterministically without any network call or API cost.
- RAG tests **do** use the real `BAAI/bge-small-en-v1.5` embedding model (so retrieval-quality assertions are genuine semantic search, not a mocked stand-in), but always against a throwaway, in-memory ChromaDB collection created per test — never the real persisted `data/chroma`. The first RAG test to run pays a one-time model-load cost of a few seconds.
- `tests/manual/` holds a few small standalone smoke-test scripts (not part of the `pytest` run) used during development to sanity-check things like retrieval quality and the order flow end-to-end against the real database.

---

## 13. Example Conversations

These are real transcripts from the actual agent, running against the real seeded database with a live LLM call — not written by hand.

**Guest browsing (sales intent, RAG-backed):**

> **Customer:** do you have anything for a home office setup, budget around $30?
>
> **QuickShelf:** I'm seeing a few accessories that could work for a home-office setup, but they're a bit above the $30 range. The most affordable option right now is a 7-in-1 USB-C hub priced at $34.99, and it's in stock. If you'd like to explore other items or adjust the budget a little, just let me know!
>
> *[intent: sales]*

**Guest tries to buy → structurally redirected to log in:**

> **Customer:** great, I'll take that
>
> **QuickShelf:** Got it! To add anything to a cart or place an order, you'll need to log in (or create an account) in the app first. Once you're signed in, just let me know and we can finish up your purchase.
>
> *[intent: sales, tool_result: `{'requires_login': True}`]*

**Logged-in customer completes a real order:**

> **Customer (logged in as Layla Ahmed):** can you order me one mechanical keyboard?
>
> **QuickShelf:** Your order for one Mechanical Keyboard has been placed! The total is $79.99 and the order is confirmed (Order #1). Let me know if there's anything else you'd like to add to your cart or purchase.
>
> *[intent: sales, tool_result: `{'order_id': 1, 'total_price': 79.99, 'status': 'confirmed'}`]*

*(Behind the scenes: `create_order` atomically decremented `Mechanical Keyboard` stock by 1, created an `Order` + `OrderItem` row, and the total is computed from the live price at the moment of purchase.)*

**Customer service question (RAG over policies, not products):**

> **Customer:** if I don't like the headphones can I send them back?
>
> **QuickShelf:** Yes—you can return the headphones as long as you do so within 30 days of delivery, and they're still unused and in their original packaging. Once we receive the item, we'll process the refund to your original payment method within about 5-7 business days.
>
> *[intent: customer_service]*

**Out-of-scope question, handled honestly instead of guessed:**

> **Customer:** what's the weather like today?
>
> **QuickShelf:** I'm sorry, but I can only help with questions about our store—like orders, shipping, returns, or product information. If you have any of those, just let me know!
>
> *[intent: customer_service]*

---

## 14. Limitations & Assumptions

- **FAQ/policy deletions made by hand-editing `app/rag/knowledge_data.py` are not reconciled.** `ingest_knowledge_base.py` reconciles (removes stale vectors for) deleted **products**, since products come from the live database and their current set of ids is always knowable. FAQs and policies added or edited through the dashboard bypass `knowledge_data.py` entirely and live only in ChromaDB, so there's nothing to reconcile there; but if you remove an entry directly from the seed file and re-run the ingestion script, its old vector is not automatically cleaned up (delete it from the dashboard instead).
- **The cart's stock check is a soft check, not a reservation.** `add_to_cart` checks stock at the moment an item is added, but doesn't reserve or lock that stock — two customers could both add the last unit of a product to their carts. The *authoritative* check happens at order creation (`create_order`), which uses an atomic conditional `UPDATE` and will correctly reject whichever order runs out of stock first. This mirrors how most real storefronts behave (a full cart doesn't guarantee availability at checkout).
- **The RAG-add id-assignment lock is in-process only.** `dashboard/routes.py` guards new FAQ/policy id assignment with a `threading.Lock`, which prevents a collision between concurrent requests handled by threads within a single Python process (true of Flask's dev server and most simple deployments). It would not prevent a collision across multiple separate worker *processes* (e.g. several Gunicorn workers with no shared lock) — not a concern for this assessment's scope, but worth calling out if deployed behind a multi-process WSGI server.
- **Messenger conversation history is kept in an in-memory dict**, keyed by `customer_id`, rather than persisted to the database. It resets if the server restarts. The web chat's history, by contrast, lives in the Flask session and survives server restarts as long as the browser session cookie is valid.
- **Authentication is intentionally minimal**, by design: email/password with hashed storage and session-based login only. No OAuth, email verification, password reset flow, two-factor authentication, or granular permission levels — reasonable for a store with exactly two roles (`customer`, `admin`) and out of scope for this assessment.
- **The agent assumes a single active conversation per customer** (one Flask session, or one Messenger PSID) rather than juggling multiple concurrent conversations for the same customer_id.
- **The LLM is treated as untrusted for identity/authorization decisions.** Every place where "which account can this action touch" matters (which tools are bound, which `customer_id` gets passed to `add_to_cart`/`create_order`) is decided in Python from the actual session/PSID, never from anything the LLM is told or infers from the conversation text — this was deliberately stress-tested against prompt-injection attempts (e.g. a guest claiming to be logged in, or a customer asking to order "for customer #4").

---

## 15. Tech Stack Summary

| Layer            | Choice                                                                 |
|-------------------|---------------------------------------------------------------------------|
| Backend framework | Flask                                                                     |
| ORM / migrations  | SQLAlchemy (Flask-SQLAlchemy) + Flask-Migrate (Alembic)                    |
| Database          | SQLite                                                                    |
| Agent framework   | LangGraph                                                                 |
| LLM               | Groq — `openai/gpt-oss-120b` (via `langchain-groq`), temperature `0.1`   |
| Embeddings        | `BAAI/bge-small-en-v1.5` (via `langchain-huggingface` / `sentence-transformers`), runs fully locally |
| Vector store      | ChromaDB (persistent, on-disk)                                            |
| Frontend          | Flask templates (Jinja) + HTML/CSS/JavaScript                            |
| Testing           | pytest                                                                    |
| Bonus integration | Facebook Messenger (Graph API + webhook)                                 |