from pathlib import Path

APP_NAME = "DataChad"
MODEL = "gpt-3.5-turbo"
PAGE_ICON = "🤖"

DATA_PATH = Path.cwd() / "data"
DEFAULT_DATA_SOURCE = "git@github.com:gustavz/DataChad.git"

AUTHENTICATION_HELP = """
Your credentials are only stored in your session state.\n
The keys are neither exposed nor made visible or stored permanently in any way.\n
Feel free to check out [the code base](https://github.com/gustavz/DataChad) to validate how things work.
"""

OPENAI_HELP = """
You can sign-up for OpenAI's API [here](https://openai.com/blog/openai-api).\n
Once you are logged in, you find the API keys [here](https://platform.openai.com/account/api-keys)
"""

ACTIVELOOP_HELP = """
You can create an ActiveLoops account (including 500GB of free database storage) [here](https://www.activeloop.ai/).\n
Once you are logged in, you find the API token [here](https://app.activeloop.ai/profile/gustavz/apitoken).\n
The organisation name is your username, or you can create new organisations [here](https://app.activeloop.ai/organization/new/create)
"""

USAGE_HELP = f"""
These are the accumulated OpenAI API usage metrics.\n
The app uses '{MODEL}' for chat and 'text-embedding-ada-002' for embeddings.\n
Learn more about OpenAI's pricing [here](https://openai.com/pricing#language-models)
"""
