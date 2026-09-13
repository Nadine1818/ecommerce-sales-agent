"""Assembles the graph: registers every node, wires the conditional edge
that routes sales vs customer_service, and compiles it into a runnable
object."""

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    classify_intent,
    customer_service_node,
    format_response,
    route_by_intent,
    sales_node,
)
from app.agent.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    # Register every node under a name 
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("sales_node", sales_node)
    graph.add_node("customer_service_node", customer_service_node)
    graph.add_node("format_response", format_response)

    # The graph's entry point, every conversation turn starts here.
    graph.set_entry_point("classify_intent")

    # Conditional edge: after classify_intent runs, LangGraph calls
    # route_by_intent(state) and uses its return value to decide which
    # node to go to next. The dict maps "what route_by_intent
    # returns" -> "which registered node name to actually run".
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "sales": "sales_node",
            "customer_service": "customer_service_node",
        },
    )

    # Both branches converge on the same formatter before ending
    graph.add_edge("sales_node", "format_response")
    graph.add_edge("customer_service_node", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# Built once, at import time, and reused for every conversation turn,
# compiling the graph is a setup cost we don't want to repeat per message.
compiled_graph = build_graph()