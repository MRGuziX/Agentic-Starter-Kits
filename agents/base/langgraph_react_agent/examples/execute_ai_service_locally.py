import os

from agents.base.langgraph_react_agent.ai_service import deployable_ai_service

from _interactive_chat import InteractiveChat


class SimpleContext:
    """Simple context object for local execution"""

    def __init__(self, payload=None):
        self.request_payload_json = payload or {}

    def get_json(self):
        return self.request_payload_json

    def get_headers(self):
        return {}



api_key = os.getenv("API_KEY", "").strip()
base_url = os.getenv("BASE_URL", "").strip()
model_id = os.getenv("MODEL_ID", "gpt-3.5-turbo").strip()

# Ensure base_url ends with /v1 if provided
if base_url and not base_url.endswith('/v1'):
    base_url = base_url.rstrip('/') + '/v1'

stream = True
context = SimpleContext()
ai_service_resp_func = deployable_ai_service(
    context=context,
    url=base_url if base_url else None,
    model_id=model_id
)[stream]


def ai_service_invoke(payload):
    context.request_payload_json = payload
    return ai_service_resp_func(context)


chat = InteractiveChat(ai_service_invoke, stream=stream)
chat.run()
