from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import os

class Chatbot():
    load_dotenv()
    loader = TextLoader('./context_poem.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    ## VECTOR Database

    # Inititalise Pinecone client
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),
                environment='gcp-starter')

    # Define index name
    index_name = 'langchain-chatbot'

    # Check index
    if index_name not in pinecone.list_indexes():
        # Create new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=768)
        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    else:
        # Index exists, link to the existing index
        docsearch = Pinecone.from_existing_index(index_name, embeddings)

    # Model

    # Define Repo ID and connect to Minstral AI
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id = repo_id,
        model_kwargs = {"temperature":0.8, "top_k":50},
        huggingfacehub_api_token = os.getenv('HUGGINGFACE_API_KEY')
    )

    # PROMPT ENGINEERING

    template = """
    You are a laureate in English Literature. These humans will ask you questions about a poem.
    Use the following piece of context to answer these questions.
    Understand the poem in the context well.
    If you don't know the answer, just say you don't know.
    Keep the answers within 4 sentences and concise.

    Context: {context}
    Question: {question}
    Answer:

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["context", "question"]
    )

    # LangChain PIPELINE for INFERENCE

    rag_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

bot = Chatbot()
input = input("Ask me anything: ")
result = bot.rag_chain.invoke(input)
print(result)
