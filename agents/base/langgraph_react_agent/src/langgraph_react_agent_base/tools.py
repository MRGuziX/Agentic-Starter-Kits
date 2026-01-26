from langchain_core.tools import tool
from pydantic import BaseModel, Field


# 1. Define the input schema explicitly
class SearchInput(BaseModel):
    query: str = Field(description="The value to search for.")


class MathInput(BaseModel):
    query: str = Field(description="The math problem to solve.")


@tool("search", args_schema=SearchInput)
def dummy_web_search(query: str) -> list[str]:
    """
    Web search tool that returns a static list of strings.
    """
    return ["RedHat"]


@tool("add", args_schema=MathInput)
def dummy_math(query: str) -> list[str]:
    """
    Math tool that returns a static list of strings.
    """
    return ["Math response"]
