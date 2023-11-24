# DataChad V3🤖

This is an app that let's you ask questions about any data source by leveraging [embeddings](https://platform.openai.com/docs/guides/embeddings), [vector databases](https://www.activeloop.ai/), [large language models](https://platform.openai.com/docs/models/gpt-3-5) and last but not least [langchains](https://github.com/hwchase17/langchain)

## How does it work?

1. Upload any `file(s)` or enter any `path` or `url` to create Knowledge Bases which can contain multiple files of any type, format and content and create Smart FAQs which are lists of curated numbered Q&As.
2. The data source or files are loaded and splitted into text document chunks
3. The text document chunks are embedded using openai or huggingface embeddings
4. The embeddings are stored as a vector dataset to activeloop's database hub
5. A langchain is created consisting of a custom selection of an LLM model (`gpt-3.5-turbo` by default), multiple vector store as knowledge bases and a single special smart FAQ vector store
6. When asking questions to the app, the chain embeds the input prompt and does a similarity search in in the provided vector stores and uses the best results as context for the LLM to generate an appropriate response
7. Finally the chat history is cached locally to enable a [ChatGPT](https://chat.openai.com/) like Q&A conversation

## Good to know
- The app only runs on `py>=3.10`!
- To run locally or deploy somewhere, execute `cp .env.template .env` and set credentials in the newly created `.env` file. Other options are manually setting of system environment variables, or storing them into `.streamlit/secrets.toml` when hosted via streamlit.
- If you have credentials set like explained above, you can just hit `submit` in the authentication without reentering your credentials in the app.
- If you run the app consider modifying the configuration in `datachad/backend/constants.py`, e.g enabling advanced options
- Your data won't load? Feel free to open an Issue or PR and contribute!
- Use previous releases like V1 or V2 for original functionality and UI

## How does it look like?

<img src="static/datachadV3.png" width="100%"/>

## TODO LIST
If you like to contribute, feel free to grab any task
- [x] Refactor utils, especially the loaders
- [x] Add option to choose model and embeddings
- [x] Enable fully local / private mode
- [x] Add option to upload multiple files to a single dataset
- [x] Decouple datachad modules from streamlit
- [x] remove all local mode and other V1 stuff
- [x] Load existing knowledge bases
- [x] Delete existing knowledge bases
- [x] Enable streaming responses
- [x] Show retrieved context
- [x] Refactor UI
- [x] Introduce smart FAQs
- [ ] Exchange downloaded file storage with tempfile
- [ ] Add user creation and login
- [ ] Add chat history per user
- [ ] Make all I/O asynchronous
- [ ] Implement FastAPI routes and backend app
- [ ] Implement a proper frontend (react or whatever)
- [ ] containerize the app
