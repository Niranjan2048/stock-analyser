import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv
from rag import FinancialChatbot

# Fix for macOS OpenMP runtime issues
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load environment variables from .env file
load_dotenv()

import os

# Disable Streamlit's file watcher for modules
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

# Fix for macOS OpenMP runtime issues (you already have this commented)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Additional environment variable to help with PyTorch/Streamlit conflicts
os.environ["PYTHONUNBUFFERED"] = "1"

# Page configuration
st.set_page_config(
    page_title="Financial Documents RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model selection
def get_model_selection():
    models = {
        "Mixtral-8x7B": "accounts/fireworks/models/mixtral-8x7b-instruct",
        "Llama-2-70B": "accounts/fireworks/models/llama-v2-70b-chat",
        "Llama-3-8B": "accounts/fireworks/models/llama-v3-8b-instruct",
        "Fireworks Claude Opus": "accounts/fireworks/models/claude-3-opus-20240229",
        "Fireworks Claude Sonnet": "accounts/fireworks/models/claude-3-sonnet-20240229"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Model",
        options=list(models.keys()),
        index=0,
        help="Select the model to use for answering questions"
    )
    
    return models[selected_model_name], selected_model_name

# Initialize chatbot with better error handling
@st.cache_resource(show_spinner=False)
def get_chatbot(model_id):
    # Get API key from environment variable - loaded from .env file
    api_key = os.environ.get("FIREWORKS_API_KEY")
    
    with st.spinner("Initializing chatbot... This may take a few minutes."):
        try:
            if not api_key:
                st.error("No Fireworks API key found in environment variables. Please check your .env file.")
                return None
                
            chatbot = FinancialChatbot()
            
            # Only force reprocess if the vector store doesn't exist or is empty
            vector_store_path = os.path.join("processed_data", "vector_store")
            force_reprocess = not os.path.exists(vector_store_path) or len(os.listdir(vector_store_path)) == 0
            
            # Initialize the chatbot with careful error checking
            init_success = chatbot.initialize(
                force_reprocess=force_reprocess,
                fireworks_api_key=api_key,
                fireworks_model=model_id
            )
            
            if not init_success:
                st.error("Chatbot initialization failed. The vector store may be empty or there was an error processing documents.")
                return None
                
            return chatbot
            
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            return None

# Helper functions for dashboard
def load_stats():
    """Load processing statistics"""
    stats_path = os.path.join("processed_data", "processing_stats.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    return None

def load_company_list():
    """Load list of companies"""
    stats = load_stats()
    if stats and "companies" in stats:
        return [{"id": k, "name": v["name"]} for k, v in stats["companies"].items()]
    return []

def create_document_stats_chart(stats):
    """Create chart of document types"""
    if not stats or "document_types" not in stats:
        return None
    
    doc_types = stats["document_types"]
    df = pd.DataFrame({
        "Type": list(doc_types.keys()),
        "Count": list(doc_types.values())
    })
    fig = px.bar(df, x="Type", y="Count", title="Document Types Distribution")
    return fig

def create_company_docs_chart(stats):
    """Create chart of documents per company"""
    if not stats or "companies" not in stats:
        return None
    
    companies = []
    docs = []
    chunks = []
    
    for company_id, data in stats["companies"].items():
        companies.append(data["name"])
        docs.append(data["documents"])
        chunks.append(data["chunks"])
    
    df = pd.DataFrame({
        "Company": companies,
        "Documents": docs,
        "Chunks": chunks
    })
    
    fig = px.bar(df, x="Company", y=["Documents", "Chunks"], 
                 title="Documents and Chunks per Company",
                 barmode="group")
    return fig

# UI Components
def sidebar():
    st.sidebar.title("Financial Documents RAG")
    
    # Display API key status
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if api_key:
        st.sidebar.success("Fireworks API key loaded from .env file")
    else:
        st.sidebar.error("No Fireworks API key found in .env file")
        st.sidebar.markdown("""
        Create a file named `.env` in the project directory with:
        ```
        FIREWORKS_API_KEY=your-api-key-here
        ```
        """)
    
    # Get model selection
    model_id, model_name = get_model_selection()
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Chatbot", "Dashboard", "Settings"])
    
    # Company filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    companies = load_company_list()
    company_names = [company["name"] for company in companies]
    
    selected_companies = st.sidebar.multiselect(
        "Filter by Company",
        options=company_names,
        default=[]
    )
    
    # Date range filter
    min_year = 2018
    max_year = datetime.now().year
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Document type filter
    doc_types = ["annual_report", "concall_transcript", "concall_notes", "concall_ppt"]
    selected_doc_types = st.sidebar.multiselect(
        "Document Types",
        options=doc_types,
        default=doc_types
    )
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application uses RAG (Retrieval-Augmented Generation) "
        "to analyze financial documents from multiple companies."
    )
    
    return {
        "page": page,
        "model_id": model_id,
        "model_name": model_name,
        "filters": {
            "companies": selected_companies,
            "year_range": year_range,
            "doc_types": selected_doc_types
        }
    }

def show_data_missing_warning():
    st.warning(
        "No data found to process. Please ensure your data is properly organized:\n\n"
        "- Check that the 'financial_data' directory exists with company folders\n"
        "- Each company folder should contain 'annual_reports' and 'concalls' folders\n"
        "- Each company folder should have a 'collection_results.json' file\n\n"
        "Click on 'Settings' to rebuild the vector store once data is in place."
    )

def chatbot_page(model_id, model_name):
    st.title("Financial Documents RAG Chatbot")
    st.markdown(
        f"Ask questions about your financial documents across multiple companies. "
        f"Currently using **{model_name}** via Fireworks.ai"
    )
    
    # Get the chatbot
    chatbot = get_chatbot(model_id)
    
    if not chatbot:
        st.warning("Please add your Fireworks API key to the .env file to continue")
        st.code("FIREWORKS_API_KEY=your-api-key-here", language="bash")
        
        # Check if data directory exists and has content
        if not os.path.exists("financial_data") or len(os.listdir("financial_data")) == 0:
            show_data_missing_warning()
            
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sample questions
    st.markdown("### Sample Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        sample_questions = [
            "What was HDFC Life's premium growth in the last fiscal year?",
            "Compare the solvency ratios of LIC and SBI Life",
            "What are the key risks mentioned by ICICI Lombard in their latest annual report?"
        ]
        
        for q in sample_questions:
            if st.button(q):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": q})
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("Thinking..."):
                            response = chatbot.ask(q)
                            st.markdown(response)
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        response = f"I encountered an error: {str(e)}"
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update UI
                st.rerun()
    
    # User input
    if prompt := st.chat_input("Ask a question about the financial documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner(f"Thinking using {model_name}..."):
                    response = chatbot.ask(prompt)
                    st.markdown(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                response = f"I encountered an error: {str(e)}"
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def dashboard_page():
    st.title("Financial Documents Dashboard")
    
    stats = load_stats()
    
    if not stats:
        st.warning("No document statistics available. Please process documents first.")
        show_data_missing_warning()
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies", stats.get("total_companies", 0))
    
    with col2:
        st.metric("Documents", stats.get("total_documents", 0))
    
    with col3:
        st.metric("Text Chunks", stats.get("total_chunks", 0))
    
    with col4:
        docs = stats.get("total_documents", 0)
        chunks = stats.get("total_chunks", 0)
        avg_chunks_per_doc = round(chunks / docs, 1) if docs > 0 else 0
        st.metric("Avg. Chunks/Doc", avg_chunks_per_doc)
    
    # Document distribution
    st.subheader("Document Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        doc_chart = create_document_stats_chart(stats)
        if doc_chart:
            st.plotly_chart(doc_chart, use_container_width=True)
        else:
            st.info("No document type information available")
    
    with col2:
        company_chart = create_company_docs_chart(stats)
        if company_chart:
            st.plotly_chart(company_chart, use_container_width=True)
        else:
            st.info("No company document information available")
    
    # Companies table
    st.subheader("Company Details")
    
    if "companies" in stats and stats["companies"]:
        company_data = []
        for company_id, data in stats["companies"].items():
            company_data.append({
                "ID": company_id,
                "Name": data["name"],
                "Documents": data["documents"],
                "Chunks": data["chunks"],
                "Avg. Chunks/Doc": round(data["chunks"] / data["documents"], 1) if data["documents"] > 0 else 0
            })
        
        df = pd.DataFrame(company_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No company details available")
    
    # Errors section
    if "errors" in stats and stats["errors"]:
        st.subheader("Processing Errors")
        with st.expander("View Processing Errors", expanded=False):
            for error in stats["errors"]:
                st.error(error)

def settings_page(model_id, model_name):
    st.title("Settings")
    
    # Check if API key is available
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        st.error("No Fireworks API key found in .env file. Please add your API key to continue.")
        st.code("FIREWORKS_API_KEY=your-api-key-here", language="bash")
        return
    
    st.subheader("Model Settings")
    st.info(f"Currently using: **{model_name}**")
    
    # Data directory status
    st.subheader("Data Directory Status")
    
    if os.path.exists("financial_data"):
        company_dirs = [d for d in os.listdir("financial_data") 
                       if os.path.isdir(os.path.join("financial_data", d))]
        if company_dirs:
            st.success(f"Found {len(company_dirs)} company directories in 'financial_data'")
        else:
            st.warning("The 'financial_data' directory exists but contains no company folders")
    else:
        st.error("The 'financial_data' directory does not exist")
    
    # Vector store status
    vector_store_path = os.path.join("processed_data", "vector_store")
    if os.path.exists(vector_store_path) and len(os.listdir(vector_store_path)) > 0:
        st.success("Vector store exists and appears to be initialized")
    else:
        st.warning("Vector store does not exist or is empty")
    
    # Rebuild Vector Store
    st.subheader("Rebuild Vector Store")
    st.warning("Rebuilding will process all documents again and may take significant time.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        rebuild_button = st.button("Rebuild Vector Store")
    
    if rebuild_button:
        with st.spinner("Rebuilding vector store... This may take several minutes."):
            try:
                chatbot = FinancialChatbot()
                init_success = chatbot.initialize(
                    force_reprocess=True,
                    fireworks_api_key=api_key,
                    fireworks_model=model_id
                )
                
                if init_success:
                    st.success("Vector store rebuilt successfully!")
                else:
                    st.error("Failed to rebuild vector store. Check logs for details.")
            except Exception as e:
                st.error(f"Error rebuilding vector store: {str(e)}")
    
    # Advanced Settings
    st.subheader("Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.number_input(
            "Chunk Size", 
            value=1000, 
            min_value=100, 
            max_value=5000, 
            step=100,
            help="Size of text chunks in characters"
        )
    
    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap", 
            value=200, 
            min_value=0, 
            max_value=1000, 
            step=50,
            help="Overlap between text chunks in characters"
        )
    
    # Model Parameters
    st.subheader("Model Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2,
            step=0.05,
            help="Higher values make output more random, lower values more deterministic"
        )
    
    with col2:
        top_k = st.number_input(
            "Retrieval Top K", 
            value=8, 
            min_value=1, 
            max_value=20, 
            step=1,
            help="Number of documents to retrieve for each query"
        )
    
    # Save settings button (not functional in this demo)
    if st.button("Save Settings"):
        st.info("Settings saved! These will take effect the next time the vector store is rebuilt.")

def main():
    # Get sidebar selections
    selections = sidebar()
    
    # Display the selected page
    if selections["page"] == "Chatbot":
        chatbot_page(selections["model_id"], selections["model_name"])
    elif selections["page"] == "Dashboard":
        dashboard_page()
    elif selections["page"] == "Settings":
        settings_page(selections["model_id"], selections["model_name"])

if __name__ == "__main__":
    main()