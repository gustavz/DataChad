PAGE_ICON = "🤖"
APP_NAME = "DataChad V3"
PROJECT_URL = "https://github.com/gustavz/DataChad"


LOCAL_MODE_DISABLED_HELP = """
This is a demo hosted with limited resources. Local Mode is not enabled.\n
To use Local Mode deploy the app on your machine of choice with `ENABLE_LOCAL_MODE` set to `True`.
"""

AUTHENTICATION_HELP = f"""
Your credentials are only stored in your session state.\n
The keys are neither exposed nor made visible or stored permanently in any way.\n
Feel free to check out [the code base]({PROJECT_URL}) to validate how things work.
"""

USAGE_HELP = f"""
These are the accumulated OpenAI API usage metrics.\n
The app uses `gpt-3.5-turbo` for chat and `text-embedding-ada-002` for embeddings.\n
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

UPLOAD_HELP = """
You can upload a single or multiple files. With each upload, all files in the batch are embedded into a single vector store.\n
**Important**: If you upload new files after you already have uploaded files, a new vector store that includes all previously uploaded files is created.
This means for each combination of uploaded files, a new vector store is created.\n
To treat your new upload independently, you need to remove the previous uploads by clicking the `X`, right next to the uploaded file name.\n
**!!! All uploaded files are removed permanently from the app after the vector stores are created !!!**
"""

DATA_TYPE_HELP = """
**Knowledge Bases** can be any number of text documents of any type, content and formatting.\n\n
**Smart FAQs** need to be single documents containing numbered FAQs.
They need to be in the format of numbers with periods followed by arbirtary text.
The next FAQ is identified by two new lines `\\n\\n` followed by the next number.
You can check if your documents are correctly formatted by using the following regex pattern:\n
`r"(?=\\n\\n\d+\.)"`. Here is an example of a correctly formatted FAQ:\n
    1. First item
    Some description here.

    1. some numbered list
    2. beloing to the first item


    2. Second item
    Another description.

    a) another list
    b) but with characters


    3. Third item
    And another one.
    - a list with dashes
    - more items
"""
