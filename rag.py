import os
import json
import logging
import glob
import time
from typing import List, Dict, Any, Optional
import re
import pandas as pd
from tqdm import tqdm
import torch
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# Import Fireworks LLM
from langchain_fireworks import ChatFireworks
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("financial_rag_builder")

class FinancialDocumentProcessor:
    """Process financial documents for RAG applications"""
    
    def __init__(self, data_dir: str = "financial_data", 
                 output_dir: str = "processed_data",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize the document processor
        
        Args:
            data_dir: Directory where financial documents are stored
            output_dir: Directory to store processed data
            chunk_size: Size of text chunks for embeddings
            chunk_overlap: Overlap between text chunks
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize metadata tracking
        self.processed_files = []
        self.stats = {
            "total_companies": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "document_types": {},
            "companies": {},
            "errors": []
        }
    
    def discover_companies(self) -> List[str]:
        """Discover all company directories in the data directory
        
        Returns:
            List of company IDs
        """
        company_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Filter out directories that don't have metadata JSON files
        valid_companies = []
        for company in company_dirs:
            metadata_path = os.path.join(self.data_dir, company, "collection_results.json")
            if os.path.exists(metadata_path):
                valid_companies.append(company)
        
        logger.info(f"Discovered {len(valid_companies)} valid company directories")
        self.stats["total_companies"] = len(valid_companies)
        return valid_companies
    
    def load_company_metadata(self, company_id: str) -> Dict:
        """Load company metadata from the collection results JSON
        
        Args:
            company_id: Company identifier
        
        Returns:
            Dictionary containing company metadata
        """
        metadata_path = os.path.join(self.data_dir, company_id, "collection_results.json")
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Initialize company stats
            if company_id not in self.stats["companies"]:
                self.stats["companies"][company_id] = {
                    "name": metadata["company"]["name"],
                    "documents": 0,
                    "chunks": 0
                }
            
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata for {company_id}: {e}")
            self.stats["errors"].append(f"Error loading metadata for {company_id}: {str(e)}")
            return None
    
    def process_annual_reports(self, company_id: str, metadata: Dict) -> List[Document]:
        """Process annual reports for a company
        
        Args:
            company_id: Company identifier
            metadata: Company metadata
        
        Returns:
            List of document chunks with metadata
        """
        documents = []
        
        if "annual_reports" not in metadata:
            logger.warning(f"No annual reports found in metadata for {company_id}")
            return documents
        
        annual_reports = metadata["annual_reports"]
        logger.info(f"Processing {len(annual_reports)} annual reports for {company_id}")
        
        for report in annual_reports:
            file_path = report.get("file_path")
            
            if not file_path or not os.path.exists(file_path):
                # Try constructing the path if it's relative
                potential_path = os.path.join(self.data_dir, company_id, "annual_reports", 
                                             os.path.basename(file_path))
                if os.path.exists(potential_path):
                    file_path = potential_path
                else:
                    logger.warning(f"Could not find annual report file: {file_path}")
                    continue
            
            try:
                # Process PDF files
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    pdf_docs = loader.load()
                    
                    # Add custom metadata to each page
                    for doc in pdf_docs:
                        doc.metadata.update({
                            "company_id": company_id,
                            "company_name": metadata["company"]["name"],
                            "document_type": "annual_report",
                            "year": report.get("year", "unknown"),
                            "source": report.get("source", "unknown"),
                            "file_name": os.path.basename(file_path)
                        })
                    
                    documents.extend(pdf_docs)
                    logger.info(f"Processed PDF: {file_path} - {len(pdf_docs)} pages")
                    
                    # Update stats
                    self.stats["total_documents"] += 1
                    self.stats["companies"][company_id]["documents"] += 1
                    
                    # Track document types
                    if "annual_report" not in self.stats["document_types"]:
                        self.stats["document_types"]["annual_report"] = 0
                    self.stats["document_types"]["annual_report"] += 1
                    
                    # Add to processed files
                    self.processed_files.append({
                        "company_id": company_id,
                        "file_path": file_path,
                        "document_type": "annual_report",
                        "year": report.get("year", "unknown"),
                        "pages": len(pdf_docs)
                    })
            except Exception as e:
                logger.error(f"Error processing annual report {file_path}: {e}")
                self.stats["errors"].append(f"Error processing annual report {file_path}: {str(e)}")
        
        return documents
    
    def process_concall_documents(self, company_id: str, metadata: Dict) -> List[Document]:
        """Process concall documents for a company
        
        Args:
            company_id: Company identifier
            metadata: Company metadata
        
        Returns:
            List of document chunks with metadata
        """
        documents = []
        
        if "concalls" not in metadata:
            logger.warning(f"No concalls found in metadata for {company_id}")
            return documents
        
        concalls = metadata["concalls"]
        logger.info(f"Processing {len(concalls)} concall sets for {company_id}")
        
        for concall in concalls:
            period = concall.get("period", "unknown")
            
            for doc_info in concall.get("documents", []):
                file_path = doc_info.get("file_path")
                doc_type = doc_info.get("type", "unknown")
                
                if not file_path or not os.path.exists(file_path):
                    # Try constructing the path if it's relative
                    potential_path = os.path.join(self.data_dir, company_id, "concalls", 
                                                doc_type, os.path.basename(file_path))
                    if os.path.exists(potential_path):
                        file_path = potential_path
                    else:
                        logger.warning(f"Could not find concall document file: {file_path}")
                        continue
                
                try:
                    # Process based on file type
                    if file_path.lower().endswith('.pdf'):
                        # PDF transcripts or presentations
                        loader = PyPDFLoader(file_path)
                        pdf_docs = loader.load()
                        
                        # Add custom metadata
                        for doc in pdf_docs:
                            doc.metadata.update({
                                "company_id": company_id,
                                "company_name": metadata["company"]["name"],
                                "document_type": f"concall_{doc_type}",
                                "period": period,
                                "file_name": os.path.basename(file_path)
                            })
                        
                        documents.extend(pdf_docs)
                        logger.info(f"Processed concall PDF: {file_path} - {len(pdf_docs)} pages")
                        
                    elif file_path.lower().endswith('.md'):
                        # Markdown files (likely concall notes)
                        loader = UnstructuredMarkdownLoader(file_path)
                        md_docs = loader.load()
                        
                        # Add custom metadata
                        for doc in md_docs:
                            doc.metadata.update({
                                "company_id": company_id,
                                "company_name": metadata["company"]["name"],
                                "document_type": f"concall_{doc_type}",
                                "period": period,
                                "file_name": os.path.basename(file_path)
                            })
                        
                        documents.extend(md_docs)
                        logger.info(f"Processed concall markdown: {file_path} - {len(md_docs)} documents")
                        
                    elif file_path.lower().endswith(('.txt', '.html')):
                        # Text transcripts
                        loader = TextLoader(file_path)
                        text_docs = loader.load()
                        
                        # Add custom metadata
                        for doc in text_docs:
                            doc.metadata.update({
                                "company_id": company_id,
                                "company_name": metadata["company"]["name"],
                                "document_type": f"concall_{doc_type}",
                                "period": period,
                                "file_name": os.path.basename(file_path)
                            })
                        
                        documents.extend(text_docs)
                        logger.info(f"Processed concall text: {file_path} - {len(text_docs)} documents")
                    
                    # Update stats
                    self.stats["total_documents"] += 1
                    self.stats["companies"][company_id]["documents"] += 1
                    
                    # Track document types
                    doc_type_key = f"concall_{doc_type}"
                    if doc_type_key not in self.stats["document_types"]:
                        self.stats["document_types"][doc_type_key] = 0
                    self.stats["document_types"][doc_type_key] += 1
                    
                    # Add to processed files
                    self.processed_files.append({
                        "company_id": company_id,
                        "file_path": file_path,
                        "document_type": doc_type_key,
                        "period": period,
                        "pages": len(documents) - sum([entry["pages"] for entry in self.processed_files])
                    })
                except Exception as e:
                    logger.error(f"Error processing concall document {file_path}: {e}")
                    self.stats["errors"].append(f"Error processing concall document {file_path}: {str(e)}")
        
        return documents
    
    def process_company(self, company_id: str) -> List[Document]:
        """Process all documents for a company
        
        Args:
            company_id: Company identifier
        
        Returns:
            List of document chunks with metadata
        """
        logger.info(f"Processing company: {company_id}")
        
        # Load company metadata
        metadata = self.load_company_metadata(company_id)
        if not metadata:
            return []
        
        all_documents = []
        
        # Process annual reports
        annual_report_docs = self.process_annual_reports(company_id, metadata)
        all_documents.extend(annual_report_docs)
        
        # Process concall documents
        concall_docs = self.process_concall_documents(company_id, metadata)
        all_documents.extend(concall_docs)
        
        logger.info(f"Processed {len(all_documents)} total documents for {company_id}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks for embedding
        
        Args:
            documents: List of documents to split
        
        Returns:
            List of document chunks
        """
        chunked_docs = self.text_splitter.split_documents(documents)
        
        # Update statistics
        self.stats["total_chunks"] += len(chunked_docs)
        
        # Update per-company stats
        for doc in chunked_docs:
            company_id = doc.metadata.get("company_id")
            if company_id in self.stats["companies"]:
                self.stats["companies"][company_id]["chunks"] += 1
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def process_all_companies(self) -> List[Document]:
        """Process all companies and their documents
        
        Returns:
            List of all document chunks with metadata
        """
        companies = self.discover_companies()
        all_chunks = []
        
        for company_id in companies:
            # Process company documents
            documents = self.process_company(company_id)
            
            # Chunk documents
            if documents:
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
        
        # Save processing stats
        with open(os.path.join(self.output_dir, "processing_stats.json"), "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        # Save processed files list
        with open(os.path.join(self.output_dir, "processed_files.json"), "w") as f:
            json.dump(self.processed_files, f, indent=2, default=str)
        
        logger.info(f"Completed processing with {len(all_chunks)} total chunks")
        return all_chunks


class FinancialRAGBuilder:
    """Build a RAG system for financial documents"""
    
    def __init__(self, 
                 processed_data_dir: str = "processed_data",
                 hf_embed_model: str = "BAAI/bge-small-en-v1.5",
                 fireworks_api_key: str = None,
                 fireworks_model: str = "accounts/fireworks/models/mixtral-8x7b-instruct"):
        """Initialize the RAG builder
        
        Args:
            processed_data_dir: Directory containing processed data
            hf_embed_model: Hugging Face embedding model to use
            fireworks_api_key: Fireworks AI API key (defaults to env var FIREWORKS_API_KEY)
            fireworks_model: Fireworks AI model to use 
        """
        self.processed_data_dir = processed_data_dir
        self.hf_embed_model = hf_embed_model
        self.fireworks_model = fireworks_model
        
        # Set Fireworks API key
        self.fireworks_api_key = fireworks_api_key or os.environ.get("FIREWORKS_API_KEY")
        if not self.fireworks_api_key:
            logger.warning("No Fireworks API key provided. Please set FIREWORKS_API_KEY environment variable")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=hf_embed_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store path
        self.vector_store_path = os.path.join(processed_data_dir, "vector_store")
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def build_vector_store(self, documents: List[Document]) -> None:
        """Build a vector store from documents
        
        Args:
            documents: List of document chunks to add to the vector store
        """
        logger.info(f"Building vector store with {len(documents)} documents")
        
        # Create a new vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.vector_store_path
        )
        
        # Persist the vector store
        self.vector_store.persist()
        logger.info(f"Vector store built and persisted to {self.vector_store_path}")
    
    def load_vector_store(self) -> bool:
        """Load an existing vector store
        
        Returns:
            True if vector store was loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading vector store from {self.vector_store_path}")
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector store loaded with {self.vector_store._collection.count()} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def setup_retriever(self, search_kwargs: Dict = None, top_k: int = 5) -> None:
        """Set up the document retriever with compression
        
        Args:
            search_kwargs: Arguments for similarity search
            top_k: Number of documents to retrieve
        """
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": top_k}
        
        # Basic retriever
        base_retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # Set up the Fireworks LLM for query expansion
        llm = ChatFireworks(
            fireworks_api_key=self.fireworks_api_key,
            model=self.fireworks_model,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.95
        )
        
        # Set up MultiQueryRetriever for query expansion
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm
        )
        
        logger.info(f"Retriever set up with search parameters {search_kwargs}")
    
    def setup_qa_chain(self) -> None:
        """Set up the question-answering chain"""
        if not self.retriever:
            raise ValueError("Retriever has not been initialized")
        
        # Set up the Fireworks LLM
        llm = ChatFireworks(
            fireworks_api_key=self.fireworks_api_key,
            model=self.fireworks_model,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.95
        )
        
        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Custom prompt template
        qa_prompt = PromptTemplate(
            template="""You are a financial analyst AI assistant specialized in analyzing financial data from annual reports and earnings calls. 
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            The financial information comes from multiple sources including annual reports and earnings call transcripts.

            Context:
            {context}

            Chat History:
            {chat_history}

            Question: {question}

            Answer:""",
            input_variables=["context", "chat_history", "question"]
        )
        
        # Set up the QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        logger.info("QA chain has been set up")
    
    def process_query(self, query: str) -> Dict:
        """Process a user query and get a response
        
        Args:
            query: User query string
        
        Returns:
            Response from the QA chain
        """
        if not self.qa_chain:
            raise ValueError("QA chain has not been initialized")
        
        try:
            start_time = time.time()
            response = self.qa_chain({"question": query})
            end_time = time.time()
            
            # Add timing information
            response["execution_time"] = end_time - start_time
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "error": str(e)
            }


class FinancialChatbot:
    """Financial document chatbot interface"""
    
    def __init__(self):
        """Initialize the chatbot"""
        # Check if processor and RAG system should be initialized
        self.processor = None
        self.rag_builder = None
        self.is_initialized = False
    
    def initialize(self, 
                  data_dir: str = "financial_data",
                  processed_data_dir: str = "processed_data",
                  force_reprocess: bool = False,
                  fireworks_api_key: str = None,
                  fireworks_model: str = "accounts/fireworks/models/mixtral-8x7b-instruct"):
        """Initialize the chatbot by processing data and building the RAG system
        
        Args:
            data_dir: Directory containing financial data
            processed_data_dir: Directory to store processed data
            force_reprocess: Whether to force reprocessing of documents
            fireworks_api_key: Fireworks AI API key
            fireworks_model: Fireworks AI model to use
        """
        logger.info("Initializing Financial Chatbot...")
        
        # Initialize document processor
        self.processor = FinancialDocumentProcessor(
            data_dir=data_dir,
            output_dir=processed_data_dir
        )
        
        # Initialize RAG builder
        self.rag_builder = FinancialRAGBuilder(
            processed_data_dir=processed_data_dir,
            fireworks_api_key=fireworks_api_key,
            fireworks_model=fireworks_model
        )
        
        # Check if vector store exists and if we should use it
        vector_store_exists = os.path.exists(self.rag_builder.vector_store_path) and \
                             len(os.listdir(self.rag_builder.vector_store_path)) > 0
        
        if vector_store_exists and not force_reprocess:
            logger.info("Existing vector store found. Loading...")
            load_success = self.rag_builder.load_vector_store()
            
            if not load_success:
                logger.warning("Failed to load existing vector store. Rebuilding...")
                force_reprocess = True
        else:
            logger.info("No vector store found or force reprocessing enabled")
            force_reprocess = True
        
        if force_reprocess:
            logger.info("Processing documents...")
            all_documents = self.processor.process_all_companies()
            
            logger.info(f"Building vector store with {len(all_documents)} document chunks")
            self.rag_builder.build_vector_store(all_documents)
        
        # Set up retriever and QA chain
        self.rag_builder.setup_retriever(top_k=8)
        self.rag_builder.setup_qa_chain()
        
        self.is_initialized = True
        logger.info("Financial Chatbot initialization complete!")
    
    def ask(self, query: str) -> str:
        """Ask a question to the chatbot
        
        Args:
            query: User query string
        
        Returns:
            Response string
        """
        if not self.is_initialized:
            return "The chatbot has not been initialized. Please call initialize() first."
        
        try:
            logger.info(f"Processing query: {query}")
            response = self.rag_builder.process_query(query)
            
            answer = response.get("answer", "I'm sorry, I couldn't generate an answer.")
            logger.info(f"Generated answer in {response.get('execution_time', 0):.2f} seconds")
            
            return answer
        except Exception as e:
            logger.error(f"Error in ask: {e}")
            return f"I encountered an error: {str(e)}"


def main():
    """Main function to initialize and demonstrate the chatbot"""
    print("Initializing Financial Documents RAG Chatbot with Fireworks AI...")
    
    # Check for Fireworks API key
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
    if not fireworks_api_key:
        print("WARNING: FIREWORKS_API_KEY environment variable not set.")
        fireworks_api_key = input("Please enter your Fireworks API key: ")
        if fireworks_api_key:
            os.environ["FIREWORKS_API_KEY"] = fireworks_api_key
        else:
            print("No API key provided. Exiting.")
            return
    
    try:
        # Initialize chatbot
        chatbot = FinancialChatbot()
        chatbot.initialize(
            force_reprocess=True,
            fireworks_api_key=fireworks_api_key,
            fireworks_model="accounts/fireworks/models/mixtral-8x7b-instruct"
        )
        
        print("\nFinancial Chatbot is ready for questions! (Type 'exit' to quit)")
        
        # Simple command line interface
        while True:
            query = input("\nYour question: ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            start_time = time.time()
            response = chatbot.ask(query)
            end_time = time.time()
            
            print(f"\nResponse (in {end_time - start_time:.2f} seconds):")
            print(response)
    
    except KeyboardInterrupt:
        print("\nExiting chatbot...")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()