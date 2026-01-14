import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add parent directory to path to allow imports
parent_dir = Path(__file__).parent.parent
examples_dir = Path(__file__).parent
src_dir = parent_dir / "src"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(examples_dir))
sys.path.insert(0, str(src_dir))

from ai_service import deployable_ai_service
from _interactive_chat import InteractiveChat


class SimpleContext:
    """Simple context object for local execution"""
    def __init__(self, payload=None):
        self.request_payload_json = payload or {}
    
    def get_json(self):
        return self.request_payload_json
    
    def get_headers(self):
        return {}


# Load environment variables from parent directory
dotenv_path = parent_dir / ".env"
if dotenv_path.is_file():
    load_dotenv(dotenv_path=dotenv_path, override=True)

# Get configuration from environment variables
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

# chat = ChatOpenAI(
#         model=model_id,
#         temperature=0.01,
#         # api_key=api_key, #not needed for local implementation
#         base_url=base_url,
#     )
#
# context = RuntimeContext(api_client=client)
# ai_service_resp_func = deployable_ai_service(context=context, **online_parameters)[stream]


def ai_service_invoke(payload):
    context.request_payload_json = payload
    return ai_service_resp_func(context)

chat = InteractiveChat(ai_service_invoke, stream=stream)
chat.run()
