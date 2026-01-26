from langchain_core.tools import tool
from pydantic import BaseModel, Field


# 1. Define the input schema explicitly
class SearchInput(BaseModel):
    query: str = Field(description="The value to search for.")


class MathInput(BaseModel):
    query: str = Field(description="The math problem to solve.")


@tool("search", parse_docstring=True)
def dummy_web_search(query: str) -> list[str]:
    """
    Search the web for information about a specific topic.

    Args:
        query: The specific text string to search for. Example: "RedHat"
    """
    return ["RedHat"]


@tool("add", args_schema=MathInput)
def dummy_math(query: str) -> list[str]:
    """
    Math tool that returns a static list of strings.
    """
    return ["Math response"]
