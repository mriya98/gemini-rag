# RAG powered LLM Bot
Use Gemini or Minstral AI to build a bot using RAG technique.

LangChain is used to build the entire pipeline and Pinecone is used to store the vector database.
A file is provided with all relevant information (context_poem.txt in this project) and is
split, vectorised, and the embeddings are stored in a vector database which is used by the LLM
to answer user questions.

## Step 1: Download dependencies
To run this file, first download the dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Set up
You need to update __.env__ file with your Pinecone API and HuggingFace Access Token. If you
don't have it, you need to create a free account on both the websites and then create an 
access token.

Update `PINECONE_API_KEY` and `HUGGINGFACE_API_KEY` in _.env_ file.

Here, I have used MinstralAI. If you want to use some other LLM that is already available on
HuggingFace, then replace the `repo_id` variable in _main.py_

## Step 3: Run
The _main.py_ is ready to run using the following code

```bash
python main.py
```
