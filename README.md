# RAG powered LLM Bot
Use Gemini or Mistral AI to build a chatbot using RAG to improve AI response.

LangChain is used to build the pipeline for LLM. The context information is stored in `context.txt`. The information from knowledge document is converted into vectors and stored in [Pinecone](https://www.pinecone.io/) (a vector database). This facilitates accurate response generation by offering a way to identify and fetch relevant information from the vector db quickly. The LLM uses this additional information to formulate the response. The web UI is developed using Streamlit.

## Step 1: Download dependencies
To run this file, first download the dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Set up
You need to update __.env__ file with your [Pinecone API](https://docs.pinecone.io/guides/get-started/quickstart) and [HuggingFace Access Token](https://huggingface.co/settings/tokens). If you don't have one, you need to create a free account on both the websites and then create an API and an access token  respectively.

Update `PINECONE_API_KEY` and `HUGGINGFACE_API_KEY` in _.env_ file.

Here, I have used MistralAI API available on HuggingFace. To use another LLM for your ChatBot, replace the `repo_id` in _main.py_ with that of the desired LLM from [HuggingFace](https://huggingface.co/models).

## Step 3: Run
After completing the authentication steps, _main.py_ is ready to run using the following code

```bash
python main.py
```

### TO-DO
1. Create a web UI frontend
2. Make it General Purpose IR from doc/pdf uploaded by user
3. Add functionality to upload and retrieve information from images
