import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA





# STEP 1: Setting up our LLM

load_dotenv()

os.environ["GOOGLE_API_KEY"]= "AIzaSyB8-mupeNy17IfCIX0E9DGm-FsjOSYO9N8"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)


st.set_page_config(page_title="Travel Guide", page_icon="🔍")
st.title("Rishab Travel Guide")

@st.cache_resource
def load_embeedings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# STEP 2: PDF Extractor

def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def extract_text_from_multiple(pdf_files):
    all_text = ""
    for pdf_file in pdf_files:
        file_text =extract_text_from_pdf(pdf_file)
        all_text += file_text
    return all_text

# STEP 3: Chunking

def split_text_into_chunks(text):
    """Split text into smaller chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )


    documents = [Document(page_content=text)]

    chunks = text_splitter.split_documents(documents)
    return chunks

# Create Vector Store

def create_vector_store(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_store")
    return vectorstore

def load_existing_vector_store(embeddings):
    if os.path.exists("faiss_store"):
        vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    
# Creating Question Answering Chain

def create_qa_chain(vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    return qa_chain

# Streamlit Initialisation

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

embeddings = load_embeedings()

existing_vectorstore = load_existing_vector_store(embeddings)
if existing_vectorstore:
    st.info("📁 Found existing vector store! You can use it or create a new one.")
    
    if st.button("🔄 Load Existing Vector Store"):
        qa_chain = create_qa_chain(existing_vectorstore)
        if qa_chain:
            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True
            st.success("✅ Existing vector store loaded!")
            st.rerun() # this is used so tsht we can refresh the sate variables



    







# STREAMLIT (FRONTEND)

uploaded_files = st.file_uploader("choose a pdf file", type="pdf", accept_multiple_files=True,help="Upload one or more files")


if uploaded_files:

    st.write('Files have been uploaded')

    if st.button('Process PDFs'):

        #Extract Text
        all_text = extract_text_from_multiple(uploaded_files)
       
        #Chunk Text
        chunks = split_text_into_chunks(all_text)

        # Create vector store
        vectorstore = create_vector_store(chunks, embeddings)
        st.write("Vector store created")

        # Create q&a chain and change State variables
        if vectorstore:
            qa_chain = create_qa_chain(vectorstore)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.processed = True
                st.success("✅ New Vector Store Loaded")
                st.rerun() # this is used so tsht we can refresh the sate variables

if st.session_state.processed:
    st.subheader("Ask Questions about the Documents") 
    question = st.text_input("Enter the Question", placeholder="What are the main topics in the document")      

    if st.button("Get Answer"):
        result = st.session_state.qa_chain({"query": question})

        answer = result["result"]
        source_docs = result["source_documents"]
        
        # Display answer
        st.subheader("🤖 Answer:")
        st.write(answer)
            
        # Display sources
        st.subheader("📖 Source Documents:")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Source {i+1}"):
                st.text_area(
                    "Content:",
                    doc.page_content,
                    height=150,
                    key=f"source_{i}"
                )
    
    st.subheader("💡 Try These Example Questions:")
    examples = [
        "What are the main topics discussed?",
        "Summarize the key findings",
        "What are the recommendations?",
        "Who are the main authors or contributors?"
    ]

    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}"):
                with st.spinner("Processing..."):
                    try:
                        result = st.session_state.qa_chain({"query": example})
                        st.write("**Answer:**", result["result"])
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


   
else:
 st.info('Pleae upload PDF files')


