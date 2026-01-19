import os
from pathlib import Path
from dotenv import load_dotenv


def load_env_from_file() -> None:
    """
    Load environment variables from .env file and add them to os.environ.
    Searches for .env file in current directory and parent directories.
    This is a shared utility function that can be used by all agent templates.
    """
    current_dir = Path.cwd()
    
    # Search in current directory and parent directories
    for search_dir in [current_dir, current_dir.parent, current_dir.parent.parent]:
        dotenv_path = search_dir / ".env"
        if dotenv_path.is_file():
            # Load .env file and add variables to os.environ
            # override=False ensures existing env vars are not overwritten
            load_dotenv(dotenv_path=dotenv_path, override=False)
            break


def get_env_var(env_key: str, default: str = None) -> str | None:
    """
    Get environment variable value.
    First checks if it exists in os.environ, if not, loads from .env file and checks again.
    This is a shared utility function that can be used by all agent templates.
    
    Args:
        env_key: The environment variable key to retrieve
        default: Default value to return if not found
        
    Returns:
        The environment variable value or default if not found
    """
    # First check if already in environment
    value = os.getenv(env_key)
    if value:
        return value.strip()
    
    # If not found, try loading from .env file
    load_env_from_file()
    
    # Check again after loading .env
    value = os.getenv(env_key)
    if value:
        return value.strip()
    
    return default
