from os import getenv

from dotenv import load_dotenv


def get_env_var(env_key: str, required: bool = True) -> str | None:
    """
    Get an environment variable. If not present, load .env file and try again.
    If failed again and required=True, raise EnvironmentError.

    :param env_key: Environment variable name
    :param required: If True, raise error when variable is not set
    :return: Environment variable value or None
    """

    value = getenv(env_key)
    if value:
        return value.strip()
    else:
        load_dotenv()
        value = getenv(env_key)
        if not value and required:
            raise EnvironmentError(f"Environment variable `{env_key}` is not set")

    return value.strip() if value else None