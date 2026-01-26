from typing import Generator
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage
from agents.base.langgraph_react_agent.src.langgraph_react_agent_base.agent import get_graph_closure
from utils import get_env_var


def deployable_ai_service(context, api_key=None, base_url=None, model_id=None):
    if not api_key: api_key = get_env_var("API_KEY")
    if not base_url: base_url = get_env_var("BASE_URL")
    if not model_id: model_id = get_env_var("MODEL_ID")

    graph = get_graph_closure(model_id=model_id, base_url=base_url)

    def get_formatted_message(resp: BaseMessage) -> dict | None:
        if isinstance(resp, ToolMessage):
            return {
                "role": "tool",
                "content": f"\n🔧 Tool Output:\n {resp.content}"
            }

        if hasattr(resp, "tool_calls") and resp.tool_calls:
            tc = resp.tool_calls[0]
            return {
                "role": "assistant",
                "content": f"🤔 I am calling tool '{tc['name']}' with args: {tc['args']}"
            }

        if resp.content:
            return {"role": "assistant", "content": resp.content}

        return None

    def convert_dict_to_message(_dict: dict) -> BaseMessage:
        role = _dict.get("role")
        content = _dict.get("content", "")
        if role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    def generate(context) -> dict:
        payload = context.get_json()
        messages = [convert_dict_to_message(m) for m in payload.get("messages", [])]
        result = graph.invoke({"messages": messages})
        final_msg = result["messages"][-1]

        return {
            "headers": {"Content-Type": "application/json"},
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_msg.content}
                }]
            }
        }

    def generate_stream(context) -> Generator[dict, None, None]:
        payload = context.get_json()
        messages = [convert_dict_to_message(m) for m in payload.get("messages", [])]

        response_stream = graph.stream(
            {"messages": messages},
            stream_mode="updates"
        )

        for update in response_stream:
            node_name = list(update.keys())[0]
            data = update[node_name]

            if "messages" in data:
                # Handle cases where multiple messages might be in one update
                msgs = data["messages"]
                if not isinstance(msgs, list):
                    msgs = [msgs]

                for msg_obj in msgs:
                    message = get_formatted_message(msg_obj)

                    # Only yield if it's a valid text message for the user
                    if message:
                        yield {
                            "choices": [{
                                "index": 0,
                                "delta": message,
                                "finish_reason": None
                            }]
                        }

    return generate, generate_stream
