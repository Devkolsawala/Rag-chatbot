import argparse
import sys
import os
from pathlib import Path
import time
import psutil
import platform

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_system import EnhancedRAGSystem, create_empty_documents_folder
except ImportError as e:
    print(f"❌ Error importing enhanced RAG system: {e}")
    print("Please ensure rag_system.py is in the same directory")
    sys.exit(1)

def print_banner():
    """Print the application banner"""
    print("\n" + "="*70)
    print("🚀 ENHANCED RAG SYSTEM - EMBEDDING GENERATION")
    print("="*70)
    print("✨ Research Papers Optimized | Advanced PDF Processing")
    print("🔬 Smart Chunking | Section-Aware | Image Detection")
    print("="*70)

def get_system_info():
    """Get system information for optimization"""
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        system = platform.system()
        
        return {
            'cpu_count': cpu_count,
            'memory_gb': round(memory.total / (1024**3), 1),
            'memory_available_gb': round(memory.available / (1024**3), 1),
            'system': system,
            'python_version': platform.python_version()
        }
    except:
        return {
            'cpu_count': 'Unknown',
            'memory_gb': 'Unknown',
            'memory_available_gb': 'Unknown',
            'system': 'Unknown',
            'python_version': platform.python_version()
        }

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'langchain': 'langchain',
        'langchain-community': 'langchain_community', 
        'langchain-huggingface': 'langchain_huggingface',
        'sentence-transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',
        'transformers': 'transformers',
        'torch': 'torch',
        'PyMuPDF': 'fitz',
        'numpy': 'numpy',
        'PIL': 'PIL',
        'psutil': 'psutil'
    }
    
    missing_packages = []
    installed_versions = {}
    
    for package_name, import_name in required_packages.items():
        try:
            if import_name == 'langchain_huggingface':
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    installed_versions[package_name] = "✅"
                except ImportError:
                    try:
                        from langchain_community.embeddings import HuggingFaceEmbeddings
                        installed_versions[package_name] = "✅ (fallback)"
                    except ImportError:
                        missing_packages.append(package_name)
            else:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'Unknown')
                installed_versions[package_name] = f"✅ v{version}"
        except ImportError:
            missing_packages.append(package_name)
    
    # Display results
    print("\n📦 Package Status:")
    for package, status in installed_versions.items():
        print(f"   {package:<25} {status}")
    
    if missing_packages:
        print(f"\n❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print(f"\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All dependencies satisfied!")
    return True

def estimate_processing_time(files, system_info):
    """Estimate processing time based on files and system specs"""
    total_size_mb = sum(f['size'] for f in files) / (1024 * 1024)
    
    # Rough estimates based on file types and system specs
    pdf_files = [f for f in files if f['type'] == 'pdf']
    txt_files = [f for f in files if f['type'] == 'txt']
    
    # Base processing rates (MB/minute)
    pdf_rate = 2.0  # PDF processing is slower
    txt_rate = 10.0  # TXT processing is faster
    
    # System adjustments
    memory_factor = min(system_info.get('memory_available_gb', 4) / 4, 2.0)
    cpu_factor = min(system_info.get('cpu_count', 4) / 4, 2.0)
    
    system_multiplier = (memory_factor + cpu_factor) / 2
    
    pdf_time = sum(f['size'] / (1024*1024) for f in pdf_files) / (pdf_rate * system_multiplier)
    txt_time = sum(f['size'] / (1024*1024) for f in txt_files) / (txt_rate * system_multiplier)
    
    # Add embedding generation time
    estimated_chunks = total_size_mb * 1.5  # Rough chunk estimate
    embedding_time = estimated_chunks / (50 * system_multiplier)  # chunks per minute
    
    total_minutes = pdf_time + txt_time + embedding_time
    
    return max(total_minutes, 0.5)  # Minimum 30 seconds

def optimize_batch_size(system_info):
    """Optimize batch size based on available memory"""
    available_memory = system_info.get('memory_available_gb', 4)
    
    if available_memory > 12:
        return 32
    elif available_memory > 8:
        return 24
    elif available_memory > 6:
        return 16
    else:
        return 8

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate enhanced embeddings for research documents")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force regeneration of embeddings even if they exist"
    )
    parser.add_argument(
        "--docs-dir", 
        type=str, 
        default=None,
        help="Directory containing documents (default: data/documents)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (recommended: all-MiniLM-L6-v2, all-mpnet-base-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (auto-optimized by default)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Get system information
    system_info = get_system_info()
    print(f"\n🖥️  System Information:")
    print(f"   OS: {system_info['system']}")
    print(f"   Python: {system_info['python_version']}")
    print(f"   CPU Cores: {system_info['cpu_count']}")
    print(f"   Memory: {system_info['memory_gb']} GB ({system_info['memory_available_gb']} GB available)")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Optimize batch size if not specified
    if args.batch_size is None:
        args.batch_size = optimize_batch_size(system_info)
        print(f"\n⚡ Optimized batch size: {args.batch_size}")
    
    # Validate model
    recommended_models = [
        "all-MiniLM-L6-v2",      # Best balance
        "all-mpnet-base-v2",     # Higher quality
        "paraphrase-MiniLM-L6-v2",  # Alternative
        "multi-qa-MiniLM-L6-cos-v1"  # Question-answering optimized
    ]
    
    if args.model not in recommended_models:
        print(f"\n⚠️  Model '{args.model}' not in recommended list:")
        for model in recommended_models:
            print(f"   • {model}")
        
        proceed = input("\nContinue anyway? (y/N): ").lower().strip()
        if proceed != 'y':
            print("Exiting...")
            sys.exit(0)
    
    try:
        # Initialize the RAG system
        print(f"\n🤖 Initializing Enhanced RAG System")
        print(f"   Model: {args.model}")
        print(f"   Batch size: {args.batch_size}")
        
        start_time = time.time()
        rag = EnhancedRAGSystem(model_name=args.model)
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f} seconds")
        
        # Set documents directory
        docs_dir = args.docs_dir or os.path.join("data", "documents")
        
        # Check documents directory
        if not os.path.exists(docs_dir):
            print(f"\n📁 Documents directory not found: {docs_dir}")
            create_dir = input("Create directory? (Y/n): ").lower().strip()
            if create_dir != 'n':
                create_empty_documents_folder()
                print("\n✅ Empty documents directory created!")
                print("📄 Please add your PDF and TXT files to the directory and run this script again.")
                sys.exit(0)
            else:
                print("❌ Cannot proceed without documents directory.")
                sys.exit(1)
        
        # Scan for documents
        print(f"\n📊 Scanning documents in: {docs_dir}")
        files = rag.scan_documents_directory(docs_dir)
        
        if not files:
            print("📄 No supported documents found")
            print("Supported formats: .txt, .pdf")
            print(f"\nPlease add documents to: {docs_dir}")
            sys.exit(0)
        
        # Analyze files
        pdf_files = [f for f in files if f['type'] == 'pdf']
        txt_files = [f for f in files if f['type'] == 'txt']
        total_size_mb = sum(f['size'] for f in files) / (1024 * 1024)
        
        print(f"\n📈 Document Analysis:")
        print(f"   📁 Total files: {len(files)}")
        if pdf_files:
            pdf_size_mb = sum(f['size'] for f in pdf_files) / (1024 * 1024)
            print(f"   📑 PDF files: {len(pdf_files)} ({pdf_size_mb:.1f} MB)")
        if txt_files:
            txt_size_mb = sum(f['size'] for f in txt_files) / (1024 * 1024)
            print(f"   📄 TXT files: {len(txt_files)} ({txt_size_mb:.1f} MB)")
        print(f"   💾 Total size: {total_size_mb:.1f} MB")
        
        # Estimate processing time
        estimated_minutes = estimate_processing_time(files, system_info)
        print(f"   ⏱️  Estimated time: {estimated_minutes:.1f} minutes")
        
        if estimated_minutes > 10:
            print(f"   ⚠️  Large dataset detected - this may take a while")
        
        # Display file details if verbose
        if args.verbose:
            print(f"\n📋 File Details:")
            for file_info in files:
                size_mb = file_info['size'] / (1024 * 1024)
                file_type = file_info['type'].upper()
                icon = "📑" if file_type == "PDF" else "📄"
                print(f"   {icon} {file_info['name']} ({file_type}, {size_mb:.1f} MB)")
        
        # Confirm processing
        print(f"\n🚀 Ready to process documents with enhanced features:")
        print(f"   🔬 Advanced PDF processing (images, tables, sections)")
        print(f"   📏 Smart chunking optimized for research papers")
        print(f"   🧠 Semantic embeddings with {args.model}")
        print(f"   💾 Efficient FAISS indexing")
        
        if not args.force:
            proceed = input(f"\nProceed with embedding generation? (Y/n): ").lower().strip()
            if proceed == 'n':
                print("Operation cancelled.")
                sys.exit(0)
        
        # Generate embeddings
        print(f"\n🔄 Processing documents...")
        print("This includes:")
        print("  1. 📄 Text extraction and cleaning")
        print("  2. 🧩 Smart chunking by sections")
        print("  3. 🧠 Embedding generation")
        print("  4. 💾 FAISS index creation")
        print("  5. 💿 Saving to disk")
        
        processing_start = time.time()
        
        try:
            rag.create_embeddings(docs_dir=docs_dir, force_recreate=args.force)
            
            processing_time = time.time() - processing_start
            
            # Success metrics
            print(f"\n🎉 SUCCESS! Embeddings generated successfully")
            print(f"   ⏱️  Total time: {processing_time:.2f} seconds")
            print(f"   📊 Document chunks: {len(rag.document_chunks)}")
            print(f"   🎯 Model used: {args.model}")
            print(f"   💾 Saved to: data/vector_store/")
            print(f"   📋 Metadata: data/enhanced_metadata.json")
            
            # Performance metrics
            if processing_time > 0:
                chunks_per_second = len(rag.document_chunks) / processing_time
                mb_per_second = total_size_mb / processing_time
                print(f"   ⚡ Performance: {chunks_per_second:.1f} chunks/sec, {mb_per_second:.1f} MB/sec")
            
            # Test retrieval system
            print(f"\n🧪 Testing retrieval system...")
            test_queries = [
                "methodology",
                "results",
                "conclusion",
                "abstract"
            ]
            
            successful_tests = 0
            for query in test_queries:
                try:
                    results = rag.retrieve_documents(query, top_k=3)
                    if results:
                        top_score = results[0][1]
                        print(f"   ✅ '{query}': {len(results)} results (score: {top_score:.3f})")
                        successful_tests += 1
                    else:
                        print(f"   ⚠️ '{query}': No results")
                except Exception as e:
                    print(f"   ❌ '{query}': Error - {e}")
            
            # Quality assessment
            print(f"\n📊 Quality Assessment:")
            test_success_rate = (successful_tests / len(test_queries)) * 100
            print(f"   Test success rate: {test_success_rate:.0f}%")
            
            if test_success_rate >= 75:
                print(f"   Quality: ✅ Excellent")
            elif test_success_rate >= 50:
                print(f"   Quality: ✅ Good")
            else:
                print(f"   Quality: ⚠️ Needs improvement")
            
            # Next steps
            print(f"\n🚀 Next Steps:")
            print(f"   1. Run: streamlit run app.py")
            print(f"   2. Start chatting with your documents!")
            print(f"   3. Upload more documents anytime")
            
            # Performance recommendations
            print(f"\n💡 Performance Tips:")
            if system_info['memory_gb'] != 'Unknown' and float(system_info['memory_gb']) < 8:
                print(f"   • Consider upgrading RAM for better performance")
            if processing_time > estimated_minutes * 60 * 1.5:
                print(f"   • Processing took longer than expected - try a smaller batch size")
            print(f"   • Current batch size ({args.batch_size}) is optimized for your system")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Operation cancelled by user")
            print("Partial processing data may have been saved")
        except Exception as e:
            print(f"\n❌ ERROR during processing:")
            print(f"   {str(e)}")
            
            if args.verbose:
                import traceback
                print(f"\nDetailed error information:")
                traceback.print_exc()
            
            # Troubleshooting suggestions
            print(f"\n🔧 Troubleshooting:")
            print(f"   1. Check document formats (PDF/TXT only)")
            print(f"   2. Ensure sufficient disk space")
            print(f"   3. Try smaller batch size: --batch-size {args.batch_size // 2}")
            print(f"   4. Check memory usage during processing")
            print(f"   5. Run with --verbose for detailed logs")
            
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing enhanced RAG system dependencies...")
    
    packages = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-huggingface>=0.0.1",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "transformers>=4.25.0",
        "torch>=1.13.0",
        "PyMuPDF>=1.23.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "psutil>=5.8.0"
    ]
    
    import subprocess
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package], 
                capture_output=True, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}:")
            print(f"   Error: {e.stderr}")
            return False
    
    print("✅ All dependencies installed successfully!")
    return True

if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install-deps":
            if install_dependencies():
                print("\n🚀 Dependencies installed. You can now run:")
                print("python generate_embeddings.py")
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("\n🤖 Enhanced RAG System - Embedding Generator")
            print("\nCommands:")
            print("  python generate_embeddings.py                    # Generate embeddings")
            print("  python generate_embeddings.py --install-deps     # Install dependencies")
            print("  python generate_embeddings.py --force           # Force regenerate")
            print("  python generate_embeddings.py --verbose         # Verbose output")
            print("  python generate_embeddings.py --help            # Show this help")
            sys.exit(0)
    
    main()