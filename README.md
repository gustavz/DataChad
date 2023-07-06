# DataChad 🤖

This is an app that let's you ask questions about any data source by leveraging [embeddings](https://platform.openai.com/docs/guides/embeddings), [vector databases](https://www.activeloop.ai/), [large language models](https://platform.openai.com/docs/models/gpt-3-5) and last but not least [langchains](https://github.com/hwchase17/langchain)

## How does it work?

1. Upload any `file(s)` or enter any `path` or `url`
2. The data source is detected and loaded into text documents
3. The text documents are embedded using openai embeddings
4. The embeddings are stored as a vector dataset to activeloop's database hub
5. A langchain is created consisting of a LLM model (`gpt-3.5-turbo` by default) and the vector store as retriever
6. When asking questions to the app, the chain embeds the input prompt and does a similarity search of in the vector store and uses the best results as context for the LLM to generate an appropriate response
7. Finally the chat history is cached locally to enable a [ChatGPT](https://chat.openai.com/) like Q&A conversation

## Good to know
- The app only runs on `py>=3.10`!
- As default context this git repository is taken so you can directly start asking question about its functionality without chosing an own data source.
- To run locally or deploy somewhere, execute `cp .env.template .env` and set credentials in the newly created `.env` file. Other options are manually setting of system environment variables, or storing them into `.streamlit/secrets.toml` when hosted via streamlit.
- If you have credentials set like explained above, you can just hit `submit` in the authentication without reentering your credentials in the app.
- To enable `Local Mode` (disabled for the demo) set `ENABLE_LOCAL_MODE` to `True` in `datachad/constants.py`. You need to have the model binaries downloaded and stored inside `./models/`
- Currently supported `Local Mode` OSS model is [GPT4all](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). To add more models update `datachad/models.py`
- If you are running `Local Mode` all your data stays locally on your machine. No API calls are made. Same with the embeddings database which stores its data to `./data/`
- Your data won't load? Feel free to open an Issue or PR and contribute!
- Yes, Chad in `DataChad` refers to the well-known [meme](https://www.google.com/search?q=chad+meme)

## How does it look like?

<img src="./datachad.png" width="100%"/>

## TODO LIST
If you like to contribute, feel free to grab any task
- [x] Refactor utils, especially the loaders
- [x] Add option to choose model and embeddings
- [x] Enable fully local / private mode
- [x] Add option to upload multiple files to a single dataset
- [x] Decouple datachad modules from streamlit
- [ ] Add user creation and login
- [ ] Load and delete existing knowledge bases
- [ ] Make all I/O asynchronous
- [ ] Implement FastAPI routes and backend app
- [ ] Implement a proper frontend (react or whatever)
- [ ] containerize the app
- [ ] deploy the app with k8s
- [ ] Add Image caption and Audio transcription support
