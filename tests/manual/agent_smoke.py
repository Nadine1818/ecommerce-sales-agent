from langchain_core.messages import HumanMessage

from app import create_app
from app.agent import compiled_graph

app = create_app()

with app.app_context():
    print("=== Sales question ===")
    result = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content="do you have any wireless mice?")],
            "customer_id": 1,
            "intent": None,
            "retrieved_context": None,
            "tool_result": None,
            "response": None,
        }
    )
    print("Intent:", result["intent"])
    print("Retrieved context:", result["retrieved_context"])
    print("Response:", result["response"])

    print("\n=== Customer service question ===")
    result2 = compiled_graph.invoke(
        {
            "messages": [HumanMessage(content="what's your return policy?")],
            "customer_id": 1,
            "intent": None,
            "retrieved_context": None,
            "tool_result": None,
            "response": None,
        }
    )
    print("Intent:", result2["intent"])
    print("Retrieved context:", result2["retrieved_context"])
    print("Response:", result2["response"])