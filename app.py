import streamlit as st
from streamlit_chat import message

from datachad.constants import APP_NAME, PAGE_ICON, UPLOAD_HELP, USAGE_HELP
from datachad.helper import (
    authentication_and_options_side_bar,
    generate_response,
    initialize_session_state,
    logger,
    update_chain,
)

# default session state variables
initialize_session_state()

# Page options and header
st.set_option("client.showErrorDetails", True)
st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON, initial_sidebar_state="expanded"
)
st.markdown(
    f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> I know all about your data!</h1>",
    unsafe_allow_html=True,
)

# Define all containers upfront to ensure app UI consistency
# container to upload files
upload_container = st.container()
# container to enter any datasource string
datasource_container = st.container()
# container to display infos stored to session state
# as it needs to be accessed from submodules
st.session_state["info_container"] = st.container()
# container for chat history
response_container = st.container()
# container for text box
text_container = st.container()

# sidebar widget with authentication and options
authentication_and_options_side_bar()

# file upload and data source input widgets
uploaded_files = upload_container.file_uploader(
    "Upload Files", accept_multiple_files=True, help=UPLOAD_HELP
)
data_source = datasource_container.text_input(
    "Enter any data source",
    placeholder="Any path or url pointing to a file or directory of files",
)

# we initialize chain after authentication is OK
# and upload and data source widgets are in place
if st.session_state["chain"] is None:
    update_chain()

# generate new chain for new data source / uploaded file
# make sure to do this only once per input / on change
if data_source and data_source != st.session_state["data_source"]:
    logger.info(f"Data source provided: '{data_source}'")
    st.session_state["data_source"] = data_source
    update_chain()

if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
    logger.info(f"Uploaded files: '{uploaded_files}'")
    st.session_state["uploaded_files"] = uploaded_files
    st.session_state["data_source"] = uploaded_files
    update_chain()


# As streamlit reruns the whole script on each change
# it is necessary to repopulate the chat containers
with text_container:
    with st.form(key="prompt_input", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        col1, col2 = st.columns([3, 1])
        submit_button = col1.form_submit_button(label="Send")
        clear_button = col2.form_submit_button("Clear Conversation")

if clear_button:
    # clear all chat related caches
    st.session_state["past"] = []
    st.session_state["generated"] = []
    st.session_state["chat_history"] = []

if submit_button and user_input:
    text_container.empty()
    output = generate_response(user_input)
    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(output)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


# Usage sidebar with total used tokens and costs
# We put this at the end to be able to show usage after the first response
with st.sidebar:
    if st.session_state["usage"]:
        st.divider()
        st.title("Usage", help=USAGE_HELP)
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens", st.session_state["usage"]["total_tokens"])
        col2.metric("Total Costs in $", st.session_state["usage"]["total_cost"])
