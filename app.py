import streamlit as st
from rag_system import RAGSystem, create_sample_knowledge_base
import os
import traceback

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None

def initialize_rag_system(force_recreate=False):
    """Initialize the RAG system with better error handling"""
    try:
        with st.spinner("Initializing RAG system... This may take a moment on first run."):
            rag = RAGSystem()
            
            # Create sample data if doesn't exist or is empty
            documents_path = "data/documents.txt"
            if not os.path.exists(documents_path) or os.path.getsize(documents_path) == 0:
                st.info("Creating sample knowledge base...")
                create_sample_knowledge_base()
                
                # Verify the file was created properly
                if not os.path.exists(documents_path):
                    raise FileNotFoundError("Failed to create documents file")
                    
                file_size = os.path.getsize(documents_path)
                if file_size == 0:
                    raise ValueError("Created documents file is empty")
                    
                st.success(f"Created knowledge base ({file_size} bytes)")
            
            # Load documents
            rag.load_documents(documents_path)
            
            # Create embeddings with force recreate option
            rag.create_embeddings(force_recreate=force_recreate)
            
            return rag, None
            
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        print(f"Error: {error_msg}")
        print("Traceback:")
        traceback.print_exc()
        return None, error_msg

def clear_cache():
    """Clear cached embeddings"""
    embeddings_file = "data/embeddings.pkl"
    if os.path.exists(embeddings_file):
        try:
            os.remove(embeddings_file)
            st.success("Cache cleared successfully!")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")
    else:
        st.info("No cache file found.")

def main():
    st.title("🤖 Retrieval-Augmented Chatbot (RAG)")
    st.markdown("Ask questions about our company policies and procedures!")
    
    # Sidebar for system info and controls
    with st.sidebar:
        st.header("System Information")
        
        if st.session_state.rag_system:
            st.info(f"""
            **Model**: all-MiniLM-L6-v2
            **Vector DB**: FAISS (CPU optimized)
            **Documents**: {len(st.session_state.rag_system.documents)}
            **Index Status**: {"✅ Ready" if st.session_state.rag_system.index else "❌ Not Ready"}
            """)
        else:
            st.warning("System not initialized")
        
        st.subheader("Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reinitialize", help="Reinitialize the system"):
                st.session_state.rag_system = None
                st.session_state.initialization_error = None
                st.rerun()
        
        with col2:
            if st.button("Force Rebuild", help="Force rebuild embeddings"):
                with st.spinner("Rebuilding embeddings..."):
                    rag, error = initialize_rag_system(force_recreate=True)
                    if rag:
                        st.session_state.rag_system = rag
                        st.session_state.initialization_error = None
                        st.success("System rebuilt successfully!")
                    else:
                        st.session_state.initialization_error = error
                st.rerun()
        
        if st.button("Clear Cache", help="Clear cached embeddings"):
            clear_cache()
        
        if st.button("Clear Chat", help="Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Initialize RAG system if needed
    if st.session_state.rag_system is None and st.session_state.initialization_error is None:
        with st.spinner("Initializing RAG system..."):
            rag, error = initialize_rag_system()
            if rag:
                st.session_state.rag_system = rag
                st.success("RAG system initialized successfully!")
                st.rerun()
            else:
                st.session_state.initialization_error = error
    
    # Show initialization error if any
    if st.session_state.initialization_error:
        st.error(f"Initialization Error: {st.session_state.initialization_error}")
        st.markdown("**Troubleshooting steps:**")
        st.markdown("1. Click 'Clear Cache' to remove corrupted cache files")
        st.markdown("2. Click 'Force Rebuild' to recreate embeddings")
        st.markdown("3. Check that you have sufficient disk space and permissions")
        return
    
    # Main interface
    if st.session_state.rag_system is None:
        st.info("Please wait while the system initializes...")
        return
    
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat")
        
        # Display chat history
        if st.session_state.chat_history:
            # Show only the last exchange to keep it clean
            for i, (query, answer) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5 exchanges
                with st.container():
                    st.markdown(f"**🤔 You:** {query}")
                    st.markdown(f"**🤖 Assistant:** {answer}")
                    if i < len(st.session_state.chat_history[-5:]) - 1:  # Don't add divider after last item
                        st.divider()
        else:
            st.info("👋 Hello! I'm here to help you with company policies and procedures. Try asking about vacation policy, remote work, benefits, or any other company topic!")
        
        # Query input
        with st.form("query_form", clear_on_submit=True):
            query = st.text_input("Ask a question:", placeholder="e.g., What is our vacation policy?", key="user_input")
            submitted = st.form_submit_button("Send", use_container_width=True)
            
            if submitted and query.strip():
                try:
                    with st.spinner("🔍 Searching knowledge base..."):
                        # Retrieve relevant documents
                        retrieved_docs = st.session_state.rag_system.retrieve_documents(query.strip(), top_k=3)
                        
                        if retrieved_docs:
                            # Generate answer
                            context_docs = [doc for doc, score in retrieved_docs]
                            answer = st.session_state.rag_system.generate_answer(query.strip(), context_docs)
                        else:
                            answer = "I couldn't find relevant information for your query. Please try rephrasing your question or asking about our company policies like vacation, remote work, benefits, etc."
                        
                        # Add to chat history
                        st.session_state.chat_history.append((query.strip(), answer))
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
                    st.error("Please try again or contact support if the issue persists.")
    
    with col2:
        st.subheader("Retrieved Documents")
        if st.session_state.chat_history:
            try:
                # Show retrieved documents for last query
                last_query = st.session_state.chat_history[-1][0]
                retrieved_docs = st.session_state.rag_system.retrieve_documents(last_query, top_k=3)
                
                if retrieved_docs:
                    for i, (doc, score) in enumerate(retrieved_docs, 1):
                        with st.expander(f"Document {i} (Score: {score:.3f})"):
                            st.text_area("", doc, height=150, key=f"doc_{i}_{len(st.session_state.chat_history)}")
                else:
                    st.info("No documents retrieved for the last query.")
                    
            except Exception as e:
                st.error(f"Error retrieving documents: {e}")
        else:
            st.info("Send a message to see retrieved documents")

if __name__ == "__main__":
    main()