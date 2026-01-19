import tomllib
from pathlib import Path
from dotenv import load_dotenv
import os

def load_dotenv_with_current_path() -> None:
    """
    Load environment variables from a `.env` file in the current working directory.
    """
    dotenv_path = Path.cwd() / ".env"

    if not dotenv_path.is_file():
        raise FileNotFoundError(".env was not found or is a directory")

    load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)


def get_from_env(env_key: str) -> str | None:
    value = os.environ.get(env_key, "").strip()
    return value

def load_config(section: str | None = None) -> dict:

    dotenv_exists = True
    try:
        load_dotenv_with_current_path()
    except FileNotFoundError:
        dotenv_exists = False

    config = tomllib.loads((Path(__file__).parent / "config.toml").read_text())
    if dotenv_exists:
        dotenv_data = {
            "apikey": get_from_env("API_KEY"),
            "url": get_from_env("BASE_URL"),
            "token": get_from_env("API_TOKEN"),
            "space_id": get_from_env("SPACE_ID"),
            "deployment_id": get_from_env("DEPLOYMENT_ID"),
            "password": get_from_env("PASSWORD"),
            "username": get_from_env("USERNAME"),
            "instance_id": get_from_env("INSTANCE_ID"),
        }
        config["deployment"].update(dotenv_data)
    if section is not None:
        return config[section]
    else:
        return config