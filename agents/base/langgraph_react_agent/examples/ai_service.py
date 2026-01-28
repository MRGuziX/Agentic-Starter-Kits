from typing import Generator
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage
from agents.base.langgraph_react_agent.src.langgraph_react_agent_base.agent import get_graph_closure


def ai_stream_service(
        context,
        base_url=None,
        model_id=None
):
    """

    :param context:
    :param base_url:
    :param model_id:
    :return:
    """
    agent = get_graph_closure(model_id=model_id, base_url=base_url)

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
        result = agent.invoke({"messages": messages})
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

        response_stream = agent.stream(
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
