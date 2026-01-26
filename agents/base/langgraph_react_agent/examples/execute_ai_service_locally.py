from _interactive_chat import InteractiveChat
from agents.base.langgraph_react_agent.examples.ai_service import deployable_ai_service
from utils import get_env_var


class SimpleContext:
    """Simple context object for local execution"""

    def __init__(self, payload=None):
        self.request_payload_json = payload or {}

    def get_json(self):
        return self.request_payload_json

    def get_headers(self):
        return {}


api_key = get_env_var("API_KEY")
if not api_key:
    raise ValueError("API_KEY is required. Please set it in environment variables or .env file")

base_url = get_env_var("BASE_URL")
if not base_url:
    raise ValueError("BASE_URL is required. Please set it in environment variables or .env file")

model_id = get_env_var("MODEL_ID")
if not model_id:
    raise ValueError("MODEL_ID is required. Please set it in environment variables or .env file")

# Ensure base_url ends with /v1 if provided
if base_url and not base_url.endswith('/v1'):
    base_url = base_url.rstrip('/') + '/v1'

stream = True
context = SimpleContext()
ai_service_resp_func = deployable_ai_service(
    context=context,
    api_key=api_key,
    base_url=base_url,
    model_id=model_id
)[stream]


def ai_service_invoke(payload):
    context.request_payload_json = payload
    return ai_service_resp_func(context)


chat = InteractiveChat(ai_service_invoke, stream=stream)
chat.run()
