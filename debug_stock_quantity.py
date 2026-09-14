from langchain_core.messages import HumanMessage
from app import create_app
from app.agent import compiled_graph

app = create_app()
with app.app_context():
    result = compiled_graph.invoke({
        'messages': [HumanMessage(content='how many wireless mice do you have in stock?')],
        'customer_id': 1, 'intent': None, 'retrieved_context': None,
        'tool_result': None, 'response': None,
    })
    print('Intent:', result['intent'])
    print()
    for i, m in enumerate(result['messages']):
        tool_calls = getattr(m, 'tool_calls', None)
        print(f'--- message {i}: {type(m).__name__} ---')
        if tool_calls:
            print('  tool_calls:', tool_calls)
        print('  content:', repr(m.content)[:300])
        print()