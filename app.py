import streamlit as st
import os
import traceback
import json
from datetime import datetime
from pathlib import Path
import time

# Import the enhanced RAG system
try:
    from rag_system import EnhancedRAGSystem, create_empty_documents_folder
except ImportError:
    st.error("Please ensure rag_system.py is in the same directory")
    st.stop()

# Page config
st.set_page_config(
    page_title="Enhanced Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern chatbot UI
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background-color: #f4f6fb;
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    /* Card-like sections */
    .main-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 25px 30px;
        margin: 20px auto;
        max-width: 1000px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #4A90E2, #6A5ACD);
        color: white;
        padding: 15px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Chat container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border-radius: 12px;
        padding: 20px;
        background: #f9fbfe;
        border: 1px solid #e0e6f1;
    }

    /* User messages */
    .user-message {
        background: #4A90E2;
        color: white;
        padding: 12px 16px;
        border-radius: 14px 14px 4px 14px;
        margin: 8px 0;
        margin-left: 20%;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
        font-size: 14px;
    }

    /* Bot messages */
    .bot-message {
        background: #eef2fb;
        color: #2c3e50;
        padding: 12px 16px;
        border-radius: 14px 14px 14px 4px;
        margin: 8px 0;
        margin-right: 20%;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
        border-left: 4px solid #4A90E2;
        font-size: 14px;
    }

    /* Message timestamp */
    .message-time {
        font-size: 11px;
        color: #777;
        margin-top: 3px;
    }

    /* Input box */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #d0d7e2 !important;
        padding: 10px !important;
        font-size: 14px !important;
    }

    textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 0.2rem rgba(74,144,226,0.25) !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(90deg, #4A90E2, #6A5ACD) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease-in-out;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #357ABD, #5A4ACD) !important;
        transform: scale(1.03);
    }

    /* Example cards */
    .source-card {
        background: #ffffff;
        padding: 14px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #4A90E2;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_info' not in st.session_state:
        st.session_state.system_info = {}
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

@st.cache_data
def load_system_info():
    """Load system information from metadata"""
    try:
        metadata_file = "data/enhanced_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return {
                'total_chunks': metadata.get('total_chunks', 0),
                'files_processed': metadata.get('files_processed', 0),
                'last_updated': metadata.get('last_updated', 'Never'),
                'model_name': metadata.get('model_name', 'all-MiniLM-L6-v2'),
                'file_sources': metadata.get('file_sources', {}),
                'system_info': metadata.get('system_info', {}),
                'embeddings_exist': os.path.exists("data/vector_store")
            }
        else:
            return {
                'total_chunks': 0,
                'files_processed': 0,
                'last_updated': 'Never',
                'model_name': 'all-MiniLM-L6-v2',
                'file_sources': {},
                'system_info': {},
                'embeddings_exist': False
            }
    except Exception as e:
        st.error(f"Error loading system info: {e}")
        return {}

def initialize_rag_system():
    """Initialize the enhanced RAG system"""
    try:
        with st.spinner("🔄 Loading Enhanced RAG system..."):
            rag = EnhancedRAGSystem()
            
            if rag.load_embeddings():
                return rag, None
            else:
                return None, "No embeddings found. Please generate embeddings first by running: python rag_system.py"
                
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {str(e)}"
        print(f"Error: {error_msg}")
        traceback.print_exc()
        return None, error_msg

def display_chat_message(message, is_user=False, timestamp=None):
    """Display a chat message with proper styling"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    if is_user:
        st.markdown(f"""
        <div style="text-align: right; margin: 10px 0;">
            <div class="user-message">
                {message}
            </div>
            <div class="message-time">You • {timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: left; margin: 10px 0;">
            <div class="bot-message">
                {message}
            </div>
            <div class="message-time">Assistant • {timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

def display_typing_indicator():
    """Display typing indicator"""
    st.markdown("""
    <div style="text-align: left; margin: 10px 0;">
        <div class="bot-message">
            <div class="typing-indicator"></div>
            <div class="typing-indicator"></div>
            <div class="typing-indicator"></div>
            Assistant is typing...
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_chat_interface():
    """Enhanced chat interface"""
    # Chat header
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white; margin-bottom: 20px;">
        <h2>🤖 Research Assistant</h2>
        <p>Ask me anything about your documents!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat history container
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            # Display chat history
            for i, (query, answer, timestamp, retrieval_info) in enumerate(st.session_state.chat_history):
                display_chat_message(query, is_user=True, timestamp=timestamp)
                display_chat_message(answer, is_user=False, timestamp=timestamp)
                
                # Show retrieval info in expander
                if retrieval_info and retrieval_info.get('docs_count', 0) > 0:
                    with st.expander(f"📊 Sources ({retrieval_info['docs_count']} documents)", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Documents Found", retrieval_info['docs_count'])
                            st.metric("Top Relevance Score", f"{retrieval_info['top_score']:.3f}")
                        with col2:
                            sources = retrieval_info.get('sources', [])
                            st.write("**Sources:**")
                            for source in sources:
                                st.write(f"• {source}")
                
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
        else:
            # Welcome message
            st.markdown("""
            <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 10px; margin: 20px 0;">
                <h3>👋 Welcome to your Research Assistant!</h3>
                <p>I can help you find information from your uploaded documents.</p>
                <p><strong>Features:</strong></p>
                <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                    <li>🔍 Advanced document search</li>
                    <li>📊 Smart relevance ranking</li>
                    <li>📄 PDF and text file support</li>
                    <li>🎯 Section-aware responses</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Query input section
    st.markdown("---")
    
    # Quick examples
    st.markdown("### 💡 Quick Examples")
    example_cols = st.columns(4)
    
    examples = [
        "What is the main research question?",
        "Summarize the methodology",
        "What are the key findings?",
        "What are the limitations?"
    ]
    
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                process_query(example)
    
    st.markdown("---")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "💭 Ask your question:",
                placeholder="e.g., What are the main findings of this research? How was the methodology designed?",
                height=100,
                key="user_query"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            submitted = st.form_submit_button("🚀 Send", use_container_width=True, type="primary")
            
            # Advanced options
            with st.expander("⚙️ Options"):
                top_k = st.slider("Max sources", 1, 10, 5)
                show_scores = st.checkbox("Show relevance scores", False)
        
        if submitted and user_input.strip():
            process_query(user_input.strip(), top_k, show_scores)

def process_query(query: str, top_k: int = 5, show_scores: bool = False):
    """Process user query and generate response"""
    if not st.session_state.rag_system:
        st.error("❌ System not initialized. Please check the setup.")
        return
    
    # Add processing state
    st.session_state.is_processing = True
    
    try:
        # Show typing indicator
        typing_placeholder = st.empty()
        with typing_placeholder:
            display_typing_indicator()
        
        # Add small delay for better UX
        time.sleep(0.5)
        
        # Retrieve documents
        retrieved_docs = st.session_state.rag_system.retrieve_documents(query, top_k=top_k)
        
        if retrieved_docs:
            # Generate answer
            answer = st.session_state.rag_system.generate_enhanced_answer(query, retrieved_docs)
            
            # Prepare retrieval info
            retrieval_info = {
                'docs_count': len(retrieved_docs),
                'top_score': retrieved_docs[0][1] if retrieved_docs else 0,
                'sources': list(set(doc[2].get('source_file', 'Unknown') for doc in retrieved_docs)),
                'show_scores': show_scores,
                'scores': [doc[1] for doc in retrieved_docs] if show_scores else []
            }
        else:
            answer = "🤔 I couldn't find relevant information in your documents. Please try rephrasing your question or check if the relevant documents are uploaded."
            retrieval_info = {'docs_count': 0, 'top_score': 0, 'sources': [], 'show_scores': False}
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        # Add to chat history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append((query, answer, timestamp, retrieval_info))
        
        # Rerun to display new message
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error processing query: {e}")
        print(f"Query processing error: {e}")
        traceback.print_exc()
    finally:
        st.session_state.is_processing = False

def display_sidebar():
    """Enhanced sidebar with system status"""
    with st.sidebar:
        # System status
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: white; border-radius: 10px; margin: 10px 0;">
            <h3>📊 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        info = st.session_state.system_info
        
        # Status indicator
        if st.session_state.rag_system and info.get('embeddings_exist'):
            st.markdown("""
            <div class="status-success">
                <strong>✅ System Ready</strong><br>
                All systems operational
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{info.get('total_chunks', 0)}</h4>
                    <p>Document Chunks</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{info.get('files_processed', 0)}</h4>
                    <p>Files Processed</p>
                </div>
                """, unsafe_allow_html=True)
            
            # File breakdown
            if info.get('file_sources'):
                st.markdown("**📁 Document Library:**")
                for filename, details in info['file_sources'].items():
                    file_icon = "📄" if details['file_type'] == 'txt' else "📑"
                    st.markdown(f"{file_icon} **{filename}** ({details['chunk_count']} chunks)")
            
            last_updated = info.get('last_updated', 'Never')[:16]
            st.caption(f"Last updated: {last_updated}")
            
        elif info.get('embeddings_exist'):
            st.markdown("""
            <div class="status-warning">
                <strong>⚠️ System Available</strong><br>
                Click 'Load System' to activate
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-error">
                <strong>❌ No Documents</strong><br>
                Please add documents and generate embeddings
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Control buttons
        st.markdown("### 🔧 Controls")
        
        if not st.session_state.rag_system and info.get('embeddings_exist'):
            if st.button("🚀 Load System", use_container_width=True, type="primary"):
                rag, error = initialize_rag_system()
                if rag:
                    st.session_state.rag_system = rag
                    st.success("✅ System loaded successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {error}")
        
        if st.session_state.rag_system:
            if st.button("🔄 Reload System", use_container_width=True):
                st.session_state.rag_system = None
                st.cache_data.clear()
                st.success("System reloaded!")
                time.sleep(0.5)
                st.rerun()
        
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        ### 📖 Instructions
        
        **To get started:**
        1. Add PDF/TXT files to `data/documents/`
        2. Run: `python rag_system.py`
        3. Click 'Load System' above
        
        **Features:**
        - 🔍 Advanced PDF processing
        - 📊 Smart document chunking
        - 🎯 Relevance scoring
        - 💡 Context-aware responses
        """)
        
        # System specs
        with st.expander("🔧 System Info"):
            st.write(f"**Model:** {info.get('model_name', 'N/A')}")
            avg_chunk = info.get('system_info', {}).get('avg_chunk_length', 0)
            st.write(f"**Avg Chunk Size:** {int(avg_chunk)} chars")
            st.write(f"**Processing:** CPU optimized")

def display_document_manager():
    """Document management interface"""
    st.markdown("""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); 
                border-radius: 10px; color: white; margin: 20px 0;">
        <h3>📁 Document Manager</h3>
    </div>
    """, unsafe_allow_html=True)
    
    docs_dir = "data/documents"
    
    if not os.path.exists(docs_dir):
        st.warning("📂 Documents directory not found")
        if st.button("📁 Create Documents Directory", type="primary"):
            create_empty_documents_folder()
            st.success("✅ Directory created! Add your PDF and TXT files.")
            st.rerun()
        return
    
    # Scan for documents
    try:
        files = []
        for file_path in Path(docs_dir).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt']:
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'type': file_path.suffix[1:].upper(),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
        
        if not files:
            st.info("📄 No documents found in the directory")
            st.markdown("""
            **To add documents:**
            1. Copy your PDF and TXT files to `data/documents/`
            2. Run `python rag_system.py` to generate embeddings
            3. Reload the system
            """)
        else:
            st.success(f"📚 Found {len(files)} document(s)")
            
            # Display files
            for file_info in files:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                
                icon = "📄" if file_info['type'] == 'TXT' else "📑"
                
                with col1:
                    st.markdown(f"{icon} **{file_info['name']}**")
                with col2:
                    st.markdown(f"`{file_info['type']}`")
                with col3:
                    st.markdown(f"{file_info['size']/1024:.1f} KB")
                with col4:
                    st.markdown(file_info['modified'].strftime("%Y-%m-%d %H:%M"))
            
            # Processing instructions
            st.markdown("---")
            st.info("""
            **Next steps:**
            1. Run `python rag_system.py` to process these documents
            2. Return here and click 'Load System'
            3. Start chatting with your documents!
            """)
            
    except Exception as e:
        st.error(f"❌ Error scanning documents: {e}")

def main():
    """Main application"""
    # Initialize
    init_session_state()
    
    # Load system info
    st.session_state.system_info = load_system_info()
    
    # Main header
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; margin-bottom: 30px;">
        <h1>🤖 Enhanced Research Assistant</h1>
        <p style="font-size: 18px;">Your AI-powered document analysis companion</p>
        <p>Advanced RAG • Smart Chunking • Research-Optimized</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    display_sidebar()
    
    # Main content
    tab1, tab2 = st.tabs(["💬 Chat", "📁 Documents"])
    
    with tab1:
        # Check system status
        if st.session_state.system_info.get('embeddings_exist'):
            if not st.session_state.rag_system:
                # Auto-initialize if embeddings exist
                with st.spinner("🔄 Initializing system..."):
                    rag, error = initialize_rag_system()
                    if rag:
                        st.session_state.rag_system = rag
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
            
            if st.session_state.rag_system:
                display_chat_interface()
            else:
                st.error("❌ Failed to initialize the system. Please check the logs.")
        else:
            # No embeddings found
            st.markdown("""
            <div style="text-align: center; padding: 40px; background: #fff3cd; border-radius: 15px; margin: 20px 0; border-left: 5px solid #ffc107;">
                <h3>⚠️ System Setup Required</h3>
                <p>No embeddings found. Please follow these steps to get started:</p>
                <ol style="text-align: left; max-width: 500px; margin: 20px auto;">
                    <li>📁 Add your PDF and TXT files to <code>data/documents/</code></li>
                    <li>⚡ Run: <code>python rag_system.py</code></li>
                    <li>🔄 Reload this page</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        display_document_manager()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666; background: #f8f9fa; border-radius: 10px;">
        <p>🤖 <strong>Enhanced Research Assistant</strong> | Built with Streamlit & LangChain</p>
        <p>Advanced RAG • Smart Document Processing • Research-Optimized</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()