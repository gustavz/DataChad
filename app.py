import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message

from constants import (
    ACTIVELOOP_HELP,
    APP_NAME,
    AUTHENTICATION_HELP,
    DEFAULT_DATA_SOURCE,
    OPENAI_HELP,
    PAGE_ICON,
)
from utils import (
    authenticate,
    build_chain_and_clear_history,
    delete_uploaded_file,
    generate_response,
    logger,
    save_uploaded_file,
)

load_dotenv()


# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)
st.markdown(
    f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> I know all about your data!</h1>",
    unsafe_allow_html=True,
)

# Initialise session state variables
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
if "data_source" not in st.session_state:
    st.session_state["data_source"] = ""
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None
if "activeloop_token" not in st.session_state:
    st.session_state["activeloop_token"] = None
if "activeloop_org_name" not in st.session_state:
    st.session_state["activeloop_org_name"] = None

# Sidebar
with st.sidebar:
    st.title("Authentication", help=AUTHENTICATION_HELP)
    with st.form("authentication"):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help=OPENAI_HELP,
            placeholder="This field is mandatory",
        )
        activeloop_token = st.text_input(
            "ActiveLoop Token",
            type="password",
            help=ACTIVELOOP_HELP,
            placeholder="Optional, using ours if empty",
        )
        activeloop_org_name = st.text_input(
            "ActiveLoop Organisation Name",
            type="password",
            help=ACTIVELOOP_HELP,
            placeholder="Optional, using ours if empty",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key, activeloop_token, activeloop_org_name)

    st.info("Learn how it works [here](https://github.com/gustavz/DataChad)")
    if not st.session_state["auth_ok"]:
        st.stop()

    clear_button = st.button("Clear Conversation", key="clear")

# the chain can only be initialized after authentication is OK
if "chain" not in st.session_state:
    build_chain_and_clear_history(DEFAULT_DATA_SOURCE)

if clear_button:
    # reset chat history
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["chat_history"] = []

# file upload and data source inputs
uploaded_file = st.file_uploader("Upload a file")
data_source = st.text_input(
    "Enter any data source",
    placeholder="Any path or url pointing to a file or directory of files",
)

# generate new chain for new data source / uploaded file
# make sure to do this only once per input / on change
if data_source and data_source != st.session_state["data_source"]:
    logger.info(f"Data source provided: '{data_source}'")
    build_chain_and_clear_history(data_source)
    st.session_state["data_source"] = data_source

if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
    logger.info(f"Uploaded file: '{uploaded_file.name}'")
    data_source = save_uploaded_file(uploaded_file)
    build_chain_and_clear_history(data_source)
    delete_uploaded_file(uploaded_file)
    st.session_state["uploaded_file"] = uploaded_file


# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key="prompt_input", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(user_input)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)


if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
