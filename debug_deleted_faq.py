from langchain_core.messages import HumanMessage

from app import create_app
from app.agent import compiled_graph
from app.rag import get_all_items, retrieve

app = create_app()

with app.app_context():
    print("=== 1. Is the warranty FAQ still in ChromaDB at all? ===")
    all_faqs = get_all_items(item_type="faq")
    for faq in all_faqs:
        print(f"  id={faq['id']}  question={faq['metadata'].get('question')}")
    if not all_faqs:
        print("  (no FAQs at all)")

    print()
    print("=== 2. What does retrieve_support_info's underlying retrieve() find for 'warranty'? ===")
    results = retrieve("Do you offer warranties on electronics?", top_k=3, item_type=["faq", "policy"])
    if not results:
        print("  (nothing found — retrieval correctly returned empty)")
    for r in results:
        print(f"  distance={r['distance']:.4f}  {r['text'][:100]}")

    print()
    print("=== 3. Full agent trace for the actual question ===")
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content="Do you offer warranties on electronics?")],
            "customer_id": 1,
            "intent": None,
            "retrieved_context": None,
            "tool_result": None,
            "response": None,
        }
    )
    print(f"Intent: {result['intent']}")
    for m in result["messages"]:
        tool_calls = getattr(m, "tool_calls", None)
        print(f"--- {type(m).__name__} ---")
        if tool_calls:
            print("  tool_calls:", tool_calls)
        print("  content:", repr(m.content)[:400])
    print()
    print("Retrieved context stored in state:", result["retrieved_context"])
    print("Final response:", result["response"])