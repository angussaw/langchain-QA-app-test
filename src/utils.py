"""Python file to serve as the backend"""

import os

from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
from langchain.chains import  RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import shutil
from src.prompts import qa_template
from tempfile import NamedTemporaryFile
import yaml

# Import config vars
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


def load_PDF(uploaded_file, collection_name: str, description: str, split_params: dict) -> dict:
    """Function to convert a loaded PDF file into vector embeddings and 
    store them on a local Chroma vector database

    Args:
        uploaded_file (_type_): PDF file to be split and converted to vector embeddings
        collection_name (str): Short title of the article that acts as a unique identifier for each Chroma vector database of a PDF file
        description (str): Short description of the article and its contents, including what the article is useful for.
        split_params (dict): Configuration parameters for splitting the PDF file into separate chunks

    Returns:
        dict: Metadata of the loaded PDF file
    """
    filename = os.path.splitext(uploaded_file.name)[0]

    with NamedTemporaryFile(dir='.', suffix='.pdf', delete=False) as f:
        try:
            f.write(uploaded_file.getbuffer())
            documents = load_and_split_doc(f.name, **split_params)
            f.close()
        finally:
            os.unlink(f.name)

    no_of_documents = len(documents)
        
    create_and_persist_vector_database(documents = documents,
                                       collection_name = collection_name)

    description = description + " Input should be a fully formed question."
    metadata = {"filename": filename,
                "collection name": collection_name,
                "description": description,
                "no_of_documents": no_of_documents}
    
    return metadata


def load_and_split_doc(filename: str, 
                       chunk_size: int, 
                       chunk_overlap: int) -> list:
    """Function to split the loaded PDF file into separate smaller chunks of text
    using a CharacterTextSplitter object

    Args:
        filename (str): String containing the name of the file
        chunk_size (int): Maximum number of characters that a chunk of text can contain
        chunk_overlap (int): Number of characters that should overlap between two adjacent chunks of text

    Returns:
        list: List containing the smaller chunks of text
    """
    loader = PyPDFLoader(filename)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   separator = " ") 

    documents = text_splitter.split_documents(documents)

    return documents


def create_and_persist_vector_database(documents: list, collection_name: str):
    """Function to convert a list of documents into vector embeddings, storing them in a Chroma vector database and 
    persisting it locally on disk

    Args:
        documents (list): List containing small chunks of text
        collection_name (str): Unique identifier for the vector database
    """

    embedding_function = HuggingFaceEmbeddings(model_name = config["HUGGING_FACE_EMBEDDING_MODEL"])

    database = Chroma.from_documents(documents,
                                     embedding = embedding_function,
                                     collection_name = collection_name,
                                     persist_directory = f"{config['DB_CHROMA_PATH']}/{collection_name}")
    database.persist()


def initialize_vector_databases(collection_names: list, fetch_k: int) -> list:
    """Function to initialize vector databases that are persisted locally on disk as vector store retrievers
    that search for k-most similar vector embeddings using cosine similarity scores

    Args:
        collection_names (list): List containing the unique identifiers of the vector databases to initialize
        fetch_k (int): k-most similar vector embeddings to search for

    Returns:
        list: List containing the initialized vector store retrievers
    """
    vector_databases = []

    embedding_function = HuggingFaceEmbeddings(model_name = config["HUGGING_FACE_EMBEDDING_MODEL"])

    for collection_name in collection_names:
        database_search = Chroma(collection_name = collection_name,
                                persist_directory=f"{config['DB_CHROMA_PATH']}/{collection_name}", 
                                embedding_function=embedding_function).as_retriever(search_type="similarity", 
                                                                                    search_kwargs={"k":fetch_k})
        
        vector_databases.append(database_search)
    
    return vector_databases


def load_retrieval_QA_chains(openai_api_key: str, temperature: float, retrievers: list) -> tuple:
    """Function that uses vector store retrievers and large language models to 
    load chains for question-answering

    Args:
        openai_api_key (str): API key to query OpenAI LLMs
        temperature (float): Scaling factor that influences the rendomness of the generated response
        retrievers (list): List containing vector store retrievers

    Returns:
        tuple: Tuple containing a list of question-answering chains and an instance of the LLM
    """
    
    chains = []

    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])

    llm = ChatOpenAI(model = config["OPENAI_MODEL"], temperature = temperature, openai_api_key = openai_api_key)
    
    for retriever in retrievers:
        chain = RetrievalQA.from_chain_type(llm = llm,
                                            chain_type = "stuff", # chain_type: specifying how the RetrievalQA should pass the chunks into LLM
                                            retriever = retriever,
                                            chain_type_kwargs={'prompt': prompt})

        chains.append(chain)

    return chains, llm


def initialize_conversational_react_agent(tool_names: list, tool_descriptions: list, chains: list, llm: ChatOpenAI) -> AgentExecutor:
    """Function to create an agent that is optimized for conversation using a chat model
    Conversation agent will have access to tools that function as question-answering chains 

    Args:
        tool_names (list): List containing the names of each tool of the conversational agent
        descriptions (list): List containing the short description of each tool of the conversational agent
        chains (list): List containing the question-answering chains which will act as tools for the conversational agent
        llm (ChatOpenAI): An OpenAI chat model

    Returns:
        AgentExecutor: An agent with a set of tools
    """
    tools = []

    for i in range(len(chains)):

        tool = Tool(name=tool_names[i],
                    func=chains[i].run,
                    description=tool_descriptions[i])
        
        tools.append(tool)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(tools, llm, 
                             agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                             verbose=True, memory=memory, handle_parsing_errors=True)
                            #  agent_kwargs={'prefix':PREFIX})
                            # 'format_instructions':FORMAT_INSTRUCTIONS,
                            # 'suffix':SUFFIX

    return agent


def remove_vector_databases(collection_names: list):
    """Function to remove vector databases that are persisted locally on disk

    Args:
        collection_names (list): List of unique identifiers for each Chroma vector database to be removed
    """
    for collection_name in collection_names:
        shutil.rmtree(f"{config['DB_CHROMA_PATH']}/{collection_name}")