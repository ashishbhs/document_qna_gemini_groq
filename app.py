import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title
st.title("📚 Ask PDF")
st.subheader("Upload your PDF document and ask questions about its content!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

if uploaded_file is not None:
    # Load the PDF document
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load documents and create embeddings
    loader = PyPDFDirectoryLoader("./")  # Load from current directory
    docs = loader.load()  # Document loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings

    st.success("PDF loaded and processed! You can now ask questions about its content.")

# Text input for questions
prompt_input = st.text_input("Enter your question about the document:")

# Button to generate answers
if st.button("Get Answer"):
    if st.session_state.vectors is None:
        st.error("Please upload a PDF document first!")
    elif not prompt_input:
        st.error("Please enter a question!")
    else:
        # Create retrieval and response chains
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Questions: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Invoke retrieval chain
        response = retrieval_chain.invoke({'input': prompt_input})
        
        # Display the answer and context
        st.write("### Answer:")
        st.write(response['answer'])

        with st.expander("Document Context Used:", expanded=True):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

# Additional information
st.markdown("""
### How It Works
1. Upload a PDF document using the file uploader.
2. Enter your question in the input box.
3. Click the "Get Answer" button to receive a context-aware answer.
4. Explore the relevant context used to generate the answer in the expander.

### Note
Make sure the PDF is not too large to ensure efficient processing.
""")
