import streamlit as st
from streamlit_chat import message

from constants import APP_NAME, DEFAULT_DATA_SOURCE, PAGE_ICON
from utils import (
    delete_uploaded_file,
    generate_response,
    save_uploaded_file,
    build_chain_and_clear_history,
    validate_keys,
)


# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(page_title=APP_NAME, page_icon=PAGE_ICON)
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


# Sidebar
with st.sidebar:
    st.title("Authentication")
    with st.form("authentication"):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        activeloop_token = st.text_input(
            "ActiveLoop Token", type="password", key="activeloop_token"
        )
        activeloop_org_name = st.text_input(
            "ActiveLoop Organisation Name", type="password", key="activeloop_org_name"
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            validate_keys(openai_key, activeloop_token, activeloop_org_name)

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
    print(f"data source provided: '{data_source}'")
    build_chain_and_clear_history(data_source)
    st.session_state["data_source"] = data_source

if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
    print(f"uploaded file: '{uploaded_file.name}'")
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
