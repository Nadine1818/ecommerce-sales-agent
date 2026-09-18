# AgentState, the data structure that flows through every node in the graph
# each node reads the fields it needs and returns updates to merge back in; LangGraph handles the merging automatically.
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Conversation history,Annotated with add_messages so LangGraph appends
    # new messages onto this list automatically, instead of replacing it.
    messages: Annotated[list, add_messages]

    # The ID of the customer who initiated the conversation. 
    # This is used to retrieve context from the database.
    # none for a guest user, or the user's id for a logged-in customer.
    customer_id: Optional[int]

    # Set by classify_intent. "sales" or "customer_service".
    intent: Optional[str]

    # Set by sales_node / customer_service_node after calling retrieve().
    # A list of dicts: {"text": ..., "metadata": ..., "distance": ...}
    retrieved_context: Optional[list]

    # Set only if sales_node calls the create_order tool.
    tool_result: Optional[dict]

    # Set by format_response, the final text sent back to the customer.
    response: Optional[str]