import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms.openai import OpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv(".env")

os.environ['OPENAI_API_KEY'] = os.getenv("openai_key")

#loading text file
with open('dataset.txt', 'r') as f:
    data = f.read()
data = data.replace('\n\n','\n')

## Text Chunking
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=300, chunk_overlap=128, length_function=len)
chunks = text_splitter.split_text(data)

def load_embeddings(embed_type='openai'):
    '''
    This function loads the embeddings based on the type of embedding model
    '''
    if embed_type=='openai':
        embeddings = OpenAIEmbeddings()
    elif embed_type=='hf':
        embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2",
                                            model_kwargs = {'device': 'cpu'},
                                            encode_kwargs = {'normalize_embeddings': False})
    
    return embeddings

def load_llm(llm_type="openai",model_id=None):
    '''
    This function loads the LLM based on the type of LLM model
    parameters:
    llm_type: type of LLM model
    model_id: id of the LLM model (applicable for ollama and hf only)s
    '''
    if llm_type=='openai':
        llm = OpenAI(temperature=0)
    
    elif llm_type=='ollama':
        llm = OllamaLLM(model=model_id)

    elif llm_type=='hf':
        llm = HuggingFacePipeline.from_model_id(
                                    model_id=model_id,
                                    task="text-generation",
                                    pipeline_kwargs={"max_new_tokens": 1024,"temperature":0.1})
    return llm

# Load embeddings
embeddings = load_embeddings("hf")

# Load LLM
llm = load_llm("ollama","phi3:3.8b")

# Create a FAISS vector store from the text chunks using the loaded embeddings
vectorStore = FAISS.from_texts(chunks, embeddings)

# Save the vector store locally
vectorStore.save_local("faiss_doc_idx")

# Load a question-answering chain with the loaded LLM, using the "refine" chain type
chain = load_qa_chain(llm, chain_type="refine")

# query string
query = ""

# Perform a similarity search in the vector store using the query
docs = vectorStore.similarity_search(query)

# Use OpenAI callback to track usage and run the chain with the retrieved documents
with get_openai_callback() as cb:
    response = chain.run(input_documents=docs, question=chain)

# Print the response from the chain
print(response)