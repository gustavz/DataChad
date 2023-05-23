from pathlib import Path

APP_NAME = "DataChad"
MODEL = "gpt-3.5-turbo"
EMBEDDINGS = "openai"
PAGE_ICON = "🤖"

PROJECT_URL = "https://github.com/gustavz/DataChad"

K = 10
FETCH_K = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
TEMPERATURE = 0.7
MAX_TOKENS = 3357
ENABLE_ADVANCED_OPTIONS = True

MODEL_N_CTX = 1000
LLAMACPP_MODEL_PATH = ""
GPT4ALL_MODEL_PATH = ""
ENABLE_LOCAL_MODELS = False

DATA_PATH = Path.cwd() / "data"
DEFAULT_DATA_SOURCE = "https://github.com/gustavz/DataChad.git"

AUTHENTICATION_HELP = f"""
Your credentials are only stored in your session state.\n
The keys are neither exposed nor made visible or stored permanently in any way.\n
Feel free to check out [the code base]({PROJECT_URL}) to validate how things work.
"""

USAGE_HELP = f"""
These are the accumulated OpenAI API usage metrics.\n
The app uses '{MODEL}' for chat and 'text-embedding-ada-002' for embeddings.\n
Learn more about OpenAI's pricing [here](https://openai.com/pricing#language-models)
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
