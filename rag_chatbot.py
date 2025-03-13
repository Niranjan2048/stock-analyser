import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# Import our chatbot
from rag import FinancialChatbot

# Page configuration
st.set_page_config(
    page_title="Financial Documents RAG",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fireworks API Key input (at top of app or in sidebar)
def get_api_key():
    api_key = st.sidebar.text_input(
        "Fireworks API Key",
        type="password",
        help="Enter your Fireworks API key. Get one at https://fireworks.ai",
        value=os.environ.get("FIREWORKS_API_KEY", "")
    )
    
    if api_key:
        os.environ["FIREWORKS_API_KEY"] = api_key
        
    return api_key

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

# Initialize chatbot
@st.cache_resource
def get_chatbot(api_key, model_id):
    if not api_key:
        st.sidebar.warning("Please enter a Fireworks API key to continue")
        return None
        
    chatbot = FinancialChatbot()
    # Only force reprocess if the vector store doesn't exist
    vector_store_path = os.path.join("processed_data", "vector_store")
    force_reprocess = not os.path.exists(vector_store_path) or len(os.listdir(vector_store_path)) == 0
    
    with st.spinner("Initializing chatbot... This may take a few minutes."):
        chatbot.initialize(
            force_reprocess=force_reprocess,
            fireworks_api_key=api_key,
            fireworks_model=model_id
        )
    return chatbot

# Helper functions for dashboard
def load_stats():
    """Load processing statistics"""
    stats_path = os.path.join("processed_data", "processing_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            return json.load(f)
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
    
    # Get Fireworks API key
    api_key = get_api_key()
    
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
        "api_key": api_key,
        "model_id": model_id,
        "model_name": model_name,
        "filters": {
            "companies": selected_companies,
            "year_range": year_range,
            "doc_types": selected_doc_types
        }
    }

def chatbot_page(api_key, model_id, model_name):
    st.title("Financial Documents Chatbot")
    st.markdown(
        f"Ask questions about your financial documents across multiple companies. "
        f"Currently using **{model_name}** via Fireworks.ai"
    )
    
    # Get the chatbot
    chatbot = get_chatbot(api_key, model_id)
    
    if not chatbot:
        st.warning("Please enter a valid Fireworks API key in the sidebar to continue")
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
                    with st.spinner("Thinking..."):
                        response = chatbot.ask(q)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update UI
                st.experimental_rerun()
    
    # User input
    if prompt := st.chat_input("Ask a question about the financial documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {model_name}..."):
                response = chatbot.ask(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def dashboard_page():
    st.title("Financial Documents Dashboard")
    
    stats = load_stats()
    
    if not stats:
        st.warning("No document statistics available. Please process documents first.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies", stats["total_companies"])
    
    with col2:
        st.metric("Documents", stats["total_documents"])
    
    with col3:
        st.metric("Text Chunks", stats["total_chunks"])
    
    with col4:
        avg_chunks_per_doc = round(stats["total_chunks"] / stats["total_documents"], 1) if stats["total_documents"] > 0 else 0
        st.metric("Avg. Chunks/Doc", avg_chunks_per_doc)
    
    # Document distribution
    st.subheader("Document Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        doc_chart = create_document_stats_chart(stats)
        if doc_chart:
            st.plotly_chart(doc_chart, use_container_width=True)
    
    with col2:
        company_chart = create_company_docs_chart(stats)
        if company_chart:
            st.plotly_chart(company_chart, use_container_width=True)
    
    # Companies table
    st.subheader("Company Details")
    
    if "companies" in stats:
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
    
    # Errors section
    if "errors" in stats and stats["errors"]:
        st.subheader("Processing Errors")
        for error in stats["errors"]:
            st.error(error)

def settings_page(api_key, model_id, model_name):
    st.title("Settings")
    
    if not api_key:
        st.warning("Please enter a Fireworks API key in the sidebar to continue")
        return
    
    st.subheader("Model Settings")
    st.info(f"Currently using: **{model_name}**")
    
    # Model parameters
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
        top_p = st.slider(
            "Top P", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.95,
            step=0.05,
            help="Controls diversity via nucleus sampling"
        )
    
    st.subheader("Rebuild Vector Store")
    st.warning("Rebuilding will process all documents again and may take significant time.")
    
    if st.button("Rebuild Vector Store"):
        with st.spinner("Rebuilding vector store... This may take several minutes."):
            chatbot = FinancialChatbot()
            chatbot.initialize(
                force_reprocess=True,
                fireworks_api_key=api_key,
                fireworks_model=model_id
            )
        st.success("Vector store rebuilt successfully!")
    
    st.subheader("Document Processing Settings")
    
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
    
    st.subheader("Retrieval Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        chatbot_page(selections["api_key"], selections["model_id"], selections["model_name"])
    elif selections["page"] == "Dashboard":
        dashboard_page()
    elif selections["page"] == "Settings":
        settings_page(selections["api_key"], selections["model_id"], selections["model_name"])

if __name__ == "__main__":
    main()