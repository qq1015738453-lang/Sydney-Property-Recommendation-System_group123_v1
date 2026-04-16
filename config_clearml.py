import os

from clearml import Task


def _get_env(name):
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


api_host = _get_env("CLEARML_API_HOST")
web_host = _get_env("CLEARML_WEB_HOST")
files_host = _get_env("CLEARML_FILES_HOST")
access_key = _get_env("CLEARML_API_ACCESS_KEY")
secret_key = _get_env("CLEARML_API_SECRET_KEY")


if all([api_host, web_host, files_host, access_key, secret_key]):
    Task.set_credentials(
        api_host=api_host,
        web_host=web_host,
        files_host=files_host,
        key=access_key,
        secret=secret_key,
    )
