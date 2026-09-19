"""Tests for the RAG layer (app/rag/ingest.py, app/rag/retrieve.py).

Uses the REAL embedding model (BAAI/bge-small-en-v1.5, already cached
locally) so retrieval-quality assertions are genuine semantic search, not
a mocked stand-in — but every test runs against a fresh, in-memory
ChromaDB collection created per test, never the real persisted
data/chroma database. That isolation matters: without it, these tests
would read and write the same knowledge base the running app/dashboard
uses, which is exactly what happened with the leftover "Race test
question A" FAQ entry found sitting in the real collection while writing
this suite.

The first test that touches the embedding model pays a one-time load
cost (several seconds); every test after that is fast since the model is
a process-wide singleton (see app/rag/store.py's lazy-singleton pattern).
"""

import importlib
import uuid

import chromadb
import pytest

from app.rag.ingest import add_or_update_item, delete_item, ingest_all
from app.rag.retrieve import get_all_items, retrieve

# app/rag/__init__.py does `from app.rag.retrieve import get_all_items,
# retrieve` — that rebinds the *attribute* `app.rag.retrieve` (the
# package's namespace) to the retrieve FUNCTION, clobbering the automatic
# submodule reference Python set up moments earlier. `import app.rag.retrieve
# as X` would silently pick up that clobbered function instead of the
# module, so importlib is used here to fetch the real submodule object
# straight out of sys.modules.
ingest_module = importlib.import_module("app.rag.ingest")
retrieve_module = importlib.import_module("app.rag.retrieve")


@pytest.fixture()
def isolated_collection(monkeypatch):
    """A fresh, empty, in-memory ChromaDB collection, swapped in for the
    real persisted one by patching the `get_collection` name as it was
    imported into ingest.py and retrieve.py specifically — patching
    app.rag.store.get_collection alone would NOT affect those modules,
    since `from app.rag.store import get_collection` binds its own local
    reference at import time, independent of the original module's
    attribute afterward."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=f"test-{uuid.uuid4().hex}")

    monkeypatch.setattr(ingest_module, "get_collection", lambda: collection)
    monkeypatch.setattr(retrieve_module, "get_collection", lambda: collection)

    return collection


# ------------------------------------------------------------- add/update

def test_add_item_then_retrieve_all_items(isolated_collection):
    add_or_update_item(
        item_id="faq-1",
        item_type="faq",
        data={"id": 1, "question": "Do you ship internationally?", "answer": "Yes, worldwide."},
    )

    items = get_all_items(item_type="faq")

    assert len(items) == 1
    assert items[0]["id"] == "faq-1"
    assert items[0]["metadata"]["type"] == "faq"
    assert items[0]["metadata"]["question"] == "Do you ship internationally?"


def test_add_or_update_item_upserts_same_id(isolated_collection):
    """Calling it twice with the same item_id must update in place, not
    create a second entry — this is what lets the dashboard's "edit"
    flow reuse the same function as "add"."""
    add_or_update_item(
        item_id="faq-1", item_type="faq", data={"id": 1, "question": "Old Q?", "answer": "Old A."}
    )
    add_or_update_item(
        item_id="faq-1", item_type="faq", data={"id": 1, "question": "New Q?", "answer": "New A."}
    )

    items = get_all_items(item_type="faq")

    assert len(items) == 1
    assert items[0]["metadata"]["question"] == "New Q?"


def test_add_item_different_ids_do_not_collide(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q1", "answer": "A1"})
    add_or_update_item(item_id="faq-2", item_type="faq", data={"id": 2, "question": "Q2", "answer": "A2"})

    assert len(get_all_items(item_type="faq")) == 2


@pytest.mark.parametrize(
    "item_type,data,expected_fragment",
    [
        (
            "product",
            {"name": "Wireless Mouse", "category_name": "Accessories", "price": 19.99, "description": "Ergonomic."},
            "Product: Wireless Mouse",
        ),
        ("faq", {"question": "Ship internationally?", "answer": "Yes."}, "Question: Ship internationally?"),
        ("policy", {"title": "Return Policy", "content": "30 days."}, "Return Policy: 30 days."),
    ],
)
def test_build_chunk_text_per_type(isolated_collection, item_type, data, expected_fragment):
    """The embedded document text must actually contain the source
    fields — otherwise semantic search would have nothing meaningful to
    match against."""
    add_or_update_item(item_id=f"{item_type}-1", item_type=item_type, data=data)

    items = get_all_items(item_type=item_type)
    assert expected_fragment in items[0]["text"]


def test_add_item_unknown_type_raises(isolated_collection):
    with pytest.raises(ValueError, match="Unknown item_type"):
        add_or_update_item(item_id="x-1", item_type="not_a_real_type", data={})


# ----------------------------------------------------------------- delete

def test_delete_existing_item_returns_true_and_removes_it(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q", "answer": "A"})

    deleted = delete_item("faq-1")

    assert deleted is True
    assert get_all_items(item_type="faq") == []


def test_delete_nonexistent_item_returns_false(isolated_collection):
    deleted = delete_item("faq-does-not-exist")
    assert deleted is False


def test_delete_one_item_does_not_affect_others(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q1", "answer": "A1"})
    add_or_update_item(item_id="faq-2", item_type="faq", data={"id": 2, "question": "Q2", "answer": "A2"})

    delete_item("faq-1")

    remaining = get_all_items(item_type="faq")
    assert len(remaining) == 1
    assert remaining[0]["id"] == "faq-2"


# ------------------------------------------------------------- ingest_all

def test_ingest_all_bulk_loads_every_item(isolated_collection):
    faqs = [
        {"id": 1, "question": "Q1?", "answer": "A1."},
        {"id": 2, "question": "Q2?", "answer": "A2."},
        {"id": 3, "question": "Q3?", "answer": "A3."},
    ]

    ingest_all(faqs, "faq")

    assert len(get_all_items(item_type="faq")) == 3


def test_ingest_all_uses_custom_id_field(isolated_collection):
    products = [{"product_id": 7, "name": "Widget", "category_name": "Misc", "price": 1.0, "description": "x"}]

    ingest_all(products, "product", id_field="product_id")

    items = get_all_items(item_type="product")
    assert items[0]["id"] == "product-7"


# --------------------------------------------------------------- retrieve

def test_retrieve_returns_top_k_shaped_results(isolated_collection):
    ingest_all(
        [
            {"id": 1, "question": "Do you ship internationally?", "answer": "Yes, most countries."},
            {"id": 2, "question": "How long does domestic shipping take?", "answer": "1-2 business days."},
            {"id": 3, "question": "Can I cancel an order?", "answer": "Within 1 hour."},
        ],
        "faq",
    )

    results = retrieve("shipping overseas", top_k=2, item_type="faq")

    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"text", "metadata", "distance"}


def test_retrieve_semantic_match_ranks_relevant_item_first(isolated_collection):
    """The actual point of RAG: a query about shipping internationally
    should rank the international-shipping FAQ above an unrelated one,
    based on real embedding similarity — not keyword overlap."""
    ingest_all(
        [
            {"id": 1, "question": "Do you ship internationally?", "answer": "Yes, we ship worldwide, 7-14 days."},
            {"id": 2, "question": "What payment methods do you accept?", "answer": "Cards, PayPal, Apple Pay."},
        ],
        "faq",
    )

    results = retrieve("Can I get a package sent to another country?", top_k=2, item_type="faq")

    assert results[0]["metadata"]["question"] == "Do you ship internationally?"
    assert results[0]["distance"] < results[1]["distance"]


def test_retrieve_filters_by_single_item_type(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Warranty?", "answer": "1 year."})
    add_or_update_item(
        item_id="product-1",
        item_type="product",
        data={"name": "Warranty Widget", "category_name": "Misc", "price": 5.0, "description": "Has a warranty."},
    )

    results = retrieve("warranty", top_k=5, item_type="faq")

    assert len(results) == 1
    assert results[0]["metadata"]["type"] == "faq"


def test_retrieve_filters_by_multiple_item_types(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Returns?", "answer": "30 days."})
    add_or_update_item(item_id="policy-1", item_type="policy", data={"id": 1, "title": "Return Policy", "content": "30 days, unused."})
    add_or_update_item(
        item_id="product-1",
        item_type="product",
        data={"name": "Returnable Widget", "category_name": "Misc", "price": 5.0, "description": "A widget."},
    )

    results = retrieve("return policy", top_k=5, item_type=["faq", "policy"])

    assert len(results) == 2
    assert {r["metadata"]["type"] for r in results} == {"faq", "policy"}


def test_retrieve_no_filter_searches_everything(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q", "answer": "A"})
    add_or_update_item(
        item_id="product-1",
        item_type="product",
        data={"name": "Widget", "category_name": "Misc", "price": 5.0, "description": "x"},
    )

    results = retrieve("anything", top_k=5, item_type=None)

    assert len(results) == 2


def test_retrieve_empty_collection_returns_empty_list(isolated_collection):
    assert retrieve("anything", top_k=3) == []


def test_retrieve_type_with_no_matches_returns_empty_list(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q", "answer": "A"})

    assert retrieve("anything", top_k=3, item_type="policy") == []


def test_retrieve_requesting_more_than_available_does_not_error(isolated_collection):
    """top_k defaults to 3, but a fresh/small knowledge base can have
    fewer matching items than that — must not crash."""
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q", "answer": "A"})

    results = retrieve("anything", top_k=10, item_type="faq")

    assert len(results) == 1


# ------------------------------------------------------------ get_all_items

def test_get_all_items_no_filter_returns_everything(isolated_collection):
    add_or_update_item(item_id="faq-1", item_type="faq", data={"id": 1, "question": "Q", "answer": "A"})
    add_or_update_item(item_id="policy-1", item_type="policy", data={"id": 1, "title": "T", "content": "C"})

    assert len(get_all_items()) == 2


def test_get_all_items_empty_collection_returns_empty_list(isolated_collection):
    assert get_all_items(item_type="faq") == []
