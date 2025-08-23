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
# st.set_page_config(
#     page_title="Enhanced Research Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.markdown(
    """
    <style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #1a1a1a;
        line-height: 1.6;
    }

    /* Modern card-like sections with glass morphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 32px 36px;
        margin: 24px auto;
        max-width: 1000px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    /* Modern section headers with refined gradients */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px 28px;
        border-radius: 16px;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
        letter-spacing: -0.02em;
    }

    /* Enhanced chat container with modern styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border-radius: 16px;
        padding: 24px;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.02);
    }

    /* Modern user messages with refined styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 20px 20px 6px 20px;
        margin: 12px 0;
        margin-left: 20%;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
        font-size: 15px;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        letter-spacing: -0.01em;
    }

    /* Enhanced bot messages with modern design */
    .bot-message {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        color: #334155;
        padding: 16px 20px;
        border-radius: 20px 20px 20px 6px;
        margin: 12px 0;
        margin-right: 20%;
        display: inline-block;
        max-width: 75%;
        word-wrap: break-word;
        border: 1px solid rgba(226, 232, 240, 0.6);
        font-size: 15px;
        font-weight: 400;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        letter-spacing: -0.01em;
    }

    /* Refined message timestamp */
    .message-time {
        font-size: 12px;
        color: #64748b;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Animated cursor color cycling */
    @keyframes cursor-color-cycle {
        0% { border-color: #ef4444; }      /* Red */
        25% { border-color: #3b82f6; }     /* Blue */
        50% { border-color: #10b981; }     /* Green */
        75% { border-color: #3b82f6; }     /* Blue */
        100% { border-color: #ef4444; }    /* Red */
    }

    /* Ambient glow animation */
    @keyframes ambient-glow {
        0% { 
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.3),
                        0 0 20px rgba(239, 68, 68, 0.2),
                        0 0 30px rgba(239, 68, 68, 0.1);
        }
        25% { 
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.3),
                        0 0 20px rgba(59, 130, 246, 0.2),
                        0 0 30px rgba(59, 130, 246, 0.1);
        }
        50% { 
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.3),
                        0 0 20px rgba(16, 185, 129, 0.2),
                        0 0 30px rgba(16, 185, 129, 0.1);
        }
        75% { 
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.3),
                        0 0 20px rgba(239, 68, 68, 0.2),
                        0 0 30px rgba(239, 68, 68, 0.1);
        }
        100% { 
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.3),
                        0 0 20px rgba(239, 68, 68, 0.2),
                        0 0 30px rgba(239, 68, 68, 0.1);
        }
    }

    /* Modern input styling with black text and animated cursor */
    textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 16px !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        background: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important; /* Black text color */
        caret-color: #ef4444; /* Red cursor by default */
    }

    /* Animated cursor when focused */
    textarea:focus {
        border-color: #667eea !important;
        background: #ffffff !important;
        color: #000000 !important; /* Keep text black when focused */
        animation: ambient-glow 2s ease-in-out infinite; /* Add ambient glow */
        caret-color: #ef4444; /* Keep cursor color animated */
    }

    /* Apply cursor animation to all text inputs */
    textarea {
        animation: cursor-color-cycle 2s infinite;
    }

    /* Enhanced buttons with modern design */
    button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25) !important;
        letter-spacing: -0.01em !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important;
    }

    /* White background for quick examples buttons */
    .quick-examples button {
        background: #ffffff !important;
        color: #334155 !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }

    .quick-examples button:hover {
        background: #f8fafc !important;
        border-color: #667eea !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }

    /* Modern example cards */
    .source-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 18px 20px;
        border-radius: 14px;
        margin: 12px 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }

    .source-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* Status indicators with modern styling */
    .status-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }

    .status-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.25);
    }

    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        text-align: center;
        margin: 16px 0;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
    }

    /* Modern metric cards */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(226, 232, 240, 0.6);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin: 8px 0;
    }

    .metric-card h4 {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 8px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-card p {
        font-size: 13px;
        color: #64748b;
        margin: 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Enhanced sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }

    /* Modern scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1e293b;
    }
    
    p, span, div {
        font-family: 'Inter', sans-serif;
        color: #334155;
    }
    
    /* Modern form elements */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: rgba(255, 255, 255, 0.9);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Ensure all input text is black */
    input, textarea, select {
        color: #000000 !important;
    }

    /* Style placeholder text */
    textarea::placeholder {
        color: #6b7280 !important;
    }
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
    # """Enhanced chat interface"""
    # st.markdown("""
    # <div style="text-align: center; padding: 28px; 
    #             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #             border-radius: 20px; color: white; margin-bottom: 28px;
    #             box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);">
    #     <h2 style="margin: 0 0 8px 0; font-size: 28px; font-weight: 700; letter-spacing: -0.02em;">🤖 Research Assistant</h2>
    #     <p style="margin: 0; font-size: 16px; opacity: 0.9; font-weight: 500;">Ask me anything about your documents!</p>
    # </div>
    # """, unsafe_allow_html=True)
        
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
            st.markdown("""
            <div style="text-align: center; padding: 48px 32px; 
                        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                        border-radius: 20px; margin: 28px 0;
                        border: 1px solid rgba(226, 232, 240, 0.8);
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);">
                <h3 style="margin: 0 0 16px 0; font-size: 24px; font-weight: 700; color: #1e293b;">👋 Welcome to your Research Assistant!</h3>
                <p style="margin: 0 0 24px 0; font-size: 16px; color: #64748b; font-weight: 500;">I can help you find information from your uploaded documents.</p>
                <p style="margin: 0 0 20px 0; font-size: 18px; font-weight: 600; color: #334155;"><strong>Features:</strong></p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 12px; font-weight: 500;">
                        🔍 Advanced document search
                    </div>
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 16px; border-radius: 12px; font-weight: 500;">
                        📊 Smart relevance ranking
                    </div>
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; padding: 16px; border-radius: 12px; font-weight: 500;">
                        📄 PDF and text file support
                    </div>
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 16px; border-radius: 12px; font-weight: 500;">
                        🎯 Section-aware responses
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Query input section
    st.markdown("---")
    
    # Quick Examples with white background
    st.markdown("### 💡 Quick Examples")
    st.markdown('<div class="quick-examples">', unsafe_allow_html=True)
    
    example_cols = st.columns(4)
    examples = [
        "What is the main research question?",
        "Summarize the methodology",
        "What are the key findings?",
        "What are the limitations?"
    ]
    
    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(f"🔍 {example}", key=f"example_{i}", use_container_width=True):
                process_query(example)
    
    st.markdown('</div>', unsafe_allow_html=True)
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
            st.markdown("""<div class="status-success"><strong>✅ System Ready</strong><br>All systems operational</div>""", unsafe_allow_html=True)
            
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
                st.markdown("**📚 Document Library:**")
                for filename, details in info['file_sources'].items():
                    file_icon = "📄" if details['file_type'] == 'txt' else "📕"
                    st.markdown(f"{file_icon} **{filename}** ({details['chunk_count']} chunks)")
            
            last_updated = info.get('last_updated', 'Never')[:16]
            st.caption(f"Last updated: {last_updated}")
            
        elif info.get('embeddings_exist'):
            st.markdown("""<div class="status-warning"><strong>⚠️ System Available</strong><br>Click 'Load System' to activate</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-error"><strong>❌ No Documents</strong><br>Please add documents and generate embeddings</div>""", unsafe_allow_html=True)
        
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
        st.markdown("""### 📖 Instructions
        
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
    <div style="text-align: center; padding: 24px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 20px; color: white; margin: 28px 0;
                box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);">
        <h3 style="margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em;">📚 Document Manager</h3>
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
            st.markdown("""**To add documents:**
            1. Copy your PDF and TXT files to `data/documents/`
            2. Run `python rag_system.py` to generate embeddings
            3. Reload the system
            """)
        else:
            st.success(f"📚 Found {len(files)} document(s)")
            
            # Display files
            for file_info in files:
                col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                
                icon = "📄" if file_info['type'] == 'TXT' else "📕"
                
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
            st.info("""**Next steps:**
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
    # st.markdown("""<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    #             border-radius: 15px; color: white; margin-bottom: 30px;">
    #     <h1>🤖 Enhanced Research Assistant</h1>
    #     <p style="font-size: 18px;">Your AI-powered document analysis companion</p>
    #     <p>Advanced RAG • Smart Chunking • Research-Optimized</p>
    # </div>""", unsafe_allow_html=True)
    
    # Sidebar
    display_sidebar()
    
    # Main content
    tab1, tab2 = st.tabs(["💬 Chat", "📚 Documents"])
    
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
            st.markdown("""<div style="text-align: center; padding: 40px; background: #fff3cd; border-radius: 15px; margin: 20px 0; border-left: 5px solid #ffc107;">
                <h3>⚠️ System Setup Required</h3>
                <p>No embeddings found. Please follow these steps to get started:</p>
                <ol style="text-align: left; max-width: 500px; margin: 20px auto;">
                    <li>📁 Add your PDF and TXT files to <code>data/documents/</code></li>
                    <li>⚡ Run: <code>python rag_system.py</code></li>
                    <li>🔄 Reload this page</li>
                </ol>
            </div>""", unsafe_allow_html=True)
    
    with tab2:
        display_document_manager()
    
    # Footer
    # st.markdown("---")
    # st.markdown("""<div style="text-align: center; padding: 20px; color: #666; background: #f8f9fa; border-radius: 10px;">
    #     <p>🤖 <strong>Enhanced Research Assistant</strong> | Built with Streamlit & LangChain</p>
    #     <p>Advanced RAG • Smart Document Processing • Research-Optimized</p>
    # </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()