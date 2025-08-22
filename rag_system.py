import numpy as np
import faiss
import pickle
import os
import json
from typing import List, Tuple, Dict, Any, Optional
import fitz  # PyMuPDF
from pathlib import Path
import hashlib
from datetime import datetime
import re
from dataclasses import dataclass
import warnings
from io import BytesIO
from PIL import Image
import base64

# LangChain imports with proper error handling
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS as LangChainFAISS
from transformers import pipeline
import torch

@dataclass
class DocumentChunk:
    """Enhanced document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class AdvancedPDFProcessor:
    """Advanced PDF processing with image extraction and table detection"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks for research papers
            chunk_overlap=200,  # More overlap for context
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Advanced PDF text extraction with image and table handling"""
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            metadata = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'images_found': 0,
                'tables_found': 0,
                'sections': []
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text blocks with position info
                blocks = page.get_text("dict")
                page_text = []
                
                # Process text blocks
                for block in blocks.get("blocks", []):
                    if "lines" in block:  # Text block
                        block_text = []
                        for line in block["lines"]:
                            line_text = []
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Check for headings based on font size
                                    font_size = span.get("size", 12)
                                    if font_size > 14:  # Likely a heading
                                        text = f"\n## {text}\n"
                                    elif font_size > 12:  # Subheading
                                        text = f"\n### {text}\n"
                                    line_text.append(text)
                            if line_text:
                                block_text.append(" ".join(line_text))
                        
                        if block_text:
                            page_text.append(" ".join(block_text))
                
                # Extract images and their context
                images = page.get_images()
                if images:
                    metadata['images_found'] += len(images)
                    page_text.append(f"\n[{len(images)} image(s) found on page {page_num + 1}]\n")
                
                # Detect tables (simple heuristic)
                tables = self._detect_tables(page)
                if tables:
                    metadata['tables_found'] += len(tables)
                    for table in tables:
                        page_text.append(f"\n[Table detected: {table}]\n")
                
                # Clean and combine page text
                page_content = "\n".join(page_text)
                page_content = self._clean_research_paper_text(page_content)
                
                if page_content.strip():
                    full_text.append(f"\n--- Page {page_num + 1} ---\n{page_content}")
            
            doc.close()
            
            combined_text = "\n".join(full_text)
            
            # Extract sections for research papers
            sections = self._extract_sections(combined_text)
            metadata['sections'] = sections
            
            return combined_text, metadata
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return "", {}
    
    def _detect_tables(self, page) -> List[str]:
        """Simple table detection based on text arrangement"""
        tables = []
        text = page.get_text()
        
        # Look for table patterns
        lines = text.split('\n')
        potential_table_lines = []
        
        for line in lines:
            # Check for multiple columns separated by spaces/tabs
            if len(line.split()) > 3 and any(char.isdigit() for char in line):
                potential_table_lines.append(line.strip())
        
        if len(potential_table_lines) > 2:
            tables.append("Numerical data table")
        
        return tables
    
    def _extract_sections(self, text: str) -> List[str]:
        """Extract common research paper sections"""
        sections = []
        common_sections = [
            r'abstract', r'introduction', r'methodology', r'method',
            r'results', r'discussion', r'conclusion', r'references',
            r'literature review', r'background', r'related work',
            r'experiments', r'evaluation', r'analysis'
        ]
        
        for section in common_sections:
            pattern = rf'\n\s*#{1,3}\s*{section}.*?\n'
            if re.search(pattern, text, re.IGNORECASE):
                sections.append(section.title())
        
        return sections
    
    def _clean_research_paper_text(self, text: str) -> str:
        """Clean research paper text with academic formatting"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        
        # Remove page numbers and footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Fix academic citations
        text = re.sub(r'\[(\d+)\]', r' [\1]', text)
        text = re.sub(r'\(([^)]+)\)', r' (\1)', text)
        
        # Fix sentence boundaries
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'\?([A-Z])', r'? \1', text)
        text = re.sub(r'!([A-Z])', r'! \1', text)
        
        # Remove artifacts
        text = re.sub(r'[^\w\s.,!?;:()\[\]"\'-]', ' ', text)
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_text_from_txt(self, txt_path: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced TXT file processing"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            cleaned_content = self._clean_research_paper_text(content)
            
            metadata = {
                'file_size': os.path.getsize(txt_path),
                'line_count': content.count('\n'),
                'word_count': len(content.split()),
                'sections': self._extract_sections(content)
            }
            
            return cleaned_content, metadata
            
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return "", {}
    
    def create_smart_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create smart chunks optimized for research papers"""
        if not text.strip():
            return []
        
        # Split by sections first if available
        sections = metadata.get('sections', [])
        if sections:
            chunks = self._chunk_by_sections(text, metadata)
        else:
            chunks = self._chunk_by_semantic_boundaries(text, metadata)
        
        return chunks
    
    def _chunk_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk text by academic sections"""
        document_chunks = []
        
        # Try to split by sections
        section_pattern = r'\n\s*#{1,3}\s*([^#\n]+)\n'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        if len(sections) > 1:
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    section_title = sections[i].strip()
                    section_content = sections[i + 1].strip()
                    
                    if len(section_content) > 50:  # Only process substantial sections
                        enhanced_metadata = metadata.copy()
                        enhanced_metadata.update({
                            'section': section_title,
                            'chunk_type': 'section',
                            'chunk_index': len(document_chunks),
                        })
                        
                        # Further split large sections
                        if len(section_content) > 1500:
                            sub_chunks = self.text_splitter.split_text(section_content)
                            for j, sub_chunk in enumerate(sub_chunks):
                                sub_metadata = enhanced_metadata.copy()
                                sub_metadata.update({
                                    'sub_chunk_index': j,
                                    'chunk_length': len(sub_chunk),
                                    'word_count': len(sub_chunk.split())
                                })
                                
                                document_chunks.append(DocumentChunk(
                                    content=f"Section: {section_title}\n\n{sub_chunk}",
                                    metadata=sub_metadata
                                ))
                        else:
                            enhanced_metadata.update({
                                'chunk_length': len(section_content),
                                'word_count': len(section_content.split())
                            })
                            
                            document_chunks.append(DocumentChunk(
                                content=f"Section: {section_title}\n\n{section_content}",
                                metadata=enhanced_metadata
                            ))
        
        # Fallback to regular chunking if section splitting didn't work
        if not document_chunks:
            document_chunks = self._chunk_by_semantic_boundaries(text, metadata)
        
        return document_chunks
    
    def _chunk_by_semantic_boundaries(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk by semantic boundaries using LangChain"""
        doc = Document(page_content=text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        document_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_metadata = chunk.metadata.copy()
            enhanced_metadata.update({
                'chunk_index': i,
                'chunk_length': len(chunk.page_content),
                'word_count': len(chunk.page_content.split()),
                'chunk_type': 'semantic'
            })
            
            document_chunks.append(DocumentChunk(
                content=chunk.page_content,
                metadata=enhanced_metadata
            ))
        
        return document_chunks

class EnhancedRAGSystem:
    """Enhanced RAG system with improved accuracy and formatting"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", data_dir: str = "data"):
        self.model_name = model_name
        self.data_dir = data_dir
        
        # Initialize embeddings model
        try:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 16}
            )
            print(f"✅ Successfully loaded embedding model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
        
        self.processor = AdvancedPDFProcessor()
        self.document_chunks: List[DocumentChunk] = []
        self.vector_store: Optional[LangChainFAISS] = None
        
        # Files for persistence
        self.embeddings_file = os.path.join(data_dir, "enhanced_embeddings.pkl")
        self.metadata_file = os.path.join(data_dir, "enhanced_metadata.json")
        self.vector_store_path = os.path.join(data_dir, "vector_store")
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize response generator
        try:
            # Use a lightweight model for response generation
            self.response_generator = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-small",
                device=-1,  # CPU
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            print("✅ Response generator loaded")
        except Exception as e:
            print(f"⚠️ Could not load response generator: {e}")
            self.response_generator = None
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate file hash for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def scan_documents_directory(self, docs_dir: str = None) -> List[Dict[str, Any]]:
        """Scan for documents with enhanced metadata"""
        if docs_dir is None:
            docs_dir = os.path.join(self.data_dir, "documents")
        
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            return []
        
        supported_extensions = ['.txt', '.pdf']
        files = []
        
        for file_path in Path(docs_dir).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    stat = file_path.stat()
                    files.append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'type': file_path.suffix[1:].lower(),
                        'size': stat.st_size,
                        'hash': self.get_file_hash(str(file_path)),
                        'modified': stat.st_mtime,
                        'relative_path': str(file_path.relative_to(docs_dir))
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return files
    
    def load_documents_from_directory(self, docs_dir: str = None) -> bool:
        """Load and process documents with enhanced processing"""
        files = self.scan_documents_directory(docs_dir)
        
        if not files:
            print("No documents found to process.")
            return False
        
        self.document_chunks = []
        
        for file_info in files:
            print(f"Processing {file_info['name']}...")
            
            file_path = file_info['path']
            file_type = file_info['type']
            
            # Extract text and metadata
            if file_type == 'txt':
                raw_text, doc_metadata = self.processor.extract_text_from_txt(file_path)
            elif file_type == 'pdf':
                raw_text, doc_metadata = self.processor.extract_text_from_pdf(file_path)
            else:
                continue
            
            if not raw_text:
                print(f"No text extracted from {file_info['name']}")
                continue
            
            # Enhanced metadata
            base_metadata = {
                'source_file': file_info['name'],
                'source_path': file_path,
                'file_type': file_type,
                'file_hash': file_info['hash'],
                'file_size': file_info['size'],
                'processed_at': datetime.now().isoformat(),
                **doc_metadata
            }
            
            # Create smart chunks
            chunks = self.processor.create_smart_chunks(raw_text, base_metadata)
            self.document_chunks.extend(chunks)
            
            print(f"  Created {len(chunks)} chunks from {file_info['name']}")
        
        print(f"Total document chunks: {len(self.document_chunks)}")
        return len(self.document_chunks) > 0
    
    def create_embeddings(self, docs_dir: str = None, force_recreate: bool = False):
        """Create embeddings with enhanced processing"""
        
        # Check if we need to recreate
        if not force_recreate and os.path.exists(self.vector_store_path):
            if self.load_embeddings():
                print("Loaded existing embeddings")
                return
        
        # Load documents
        if not self.load_documents_from_directory(docs_dir):
            raise ValueError("Failed to load documents")
        
        print(f"Creating embeddings for {len(self.document_chunks)} chunks...")
        
        # Prepare documents for LangChain
        documents = []
        for chunk in self.document_chunks:
            doc = Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            documents.append(doc)
        
        # Create vector store with batching
        print("Building FAISS vector store...")
        try:
            # Process in batches to avoid memory issues
            batch_size = 100
            if len(documents) > batch_size:
                # Initialize with first batch
                first_batch = documents[:batch_size]
                self.vector_store = LangChainFAISS.from_documents(
                    first_batch, 
                    self.embeddings_model
                )
                
                # Add remaining documents in batches
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    batch_store = LangChainFAISS.from_documents(
                        batch,
                        self.embeddings_model
                    )
                    self.vector_store.merge_from(batch_store)
                    print(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            else:
                self.vector_store = LangChainFAISS.from_documents(
                    documents, 
                    self.embeddings_model
                )
            
            print("✅ Vector store created successfully")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
        
        # Save everything
        self.save_embeddings()
        self.save_metadata()
        
        print("✅ Enhanced embeddings created successfully!")
    
    def save_embeddings(self):
        """Save the vector store and chunks"""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        try:
            # Save vector store
            self.vector_store.save_local(self.vector_store_path)
            
            # Save document chunks
            chunks_data = {
                'chunks': [(chunk.content, chunk.metadata) for chunk in self.document_chunks],
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(chunks_data, f)
            
            print(f"✅ Embeddings saved to {self.vector_store_path}")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self) -> bool:
        """Load existing embeddings"""
        if not os.path.exists(self.vector_store_path) or not os.path.exists(self.embeddings_file):
            return False
        
        try:
            # Load vector store
            self.vector_store = LangChainFAISS.load_local(
                self.vector_store_path, 
                self.embeddings_model,
                allow_dangerous_deserialization=True
            )
            
            # Load chunks
            with open(self.embeddings_file, 'rb') as f:
                chunks_data = pickle.load(f)
            
            self.document_chunks = []
            for content, metadata in chunks_data['chunks']:
                self.document_chunks.append(DocumentChunk(content=content, metadata=metadata))
            
            print(f"✅ Loaded {len(self.document_chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def save_metadata(self):
        """Save enhanced metadata"""
        try:
            file_sources = {}
            for chunk in self.document_chunks:
                source = chunk.metadata.get('source_file', 'unknown')
                if source not in file_sources:
                    file_sources[source] = {
                        'chunk_count': 0,
                        'file_type': chunk.metadata.get('file_type', 'unknown'),
                        'file_size': chunk.metadata.get('file_size', 0),
                        'processed_at': chunk.metadata.get('processed_at', ''),
                        'sections': chunk.metadata.get('sections', [])
                    }
                file_sources[source]['chunk_count'] += 1
            
            metadata = {
                'total_chunks': len(self.document_chunks),
                'files_processed': len(file_sources),
                'file_sources': file_sources,
                'model_name': self.model_name,
                'last_updated': datetime.now().isoformat(),
                'system_info': {
                    'chunk_size_range': self._get_chunk_size_stats(),
                    'avg_chunk_length': np.mean([len(chunk.content) for chunk in self.document_chunks]) if self.document_chunks else 0
                }
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            print("✅ Metadata saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
    
    def _get_chunk_size_stats(self) -> Dict[str, int]:
        """Get chunk size statistics"""
        if not self.document_chunks:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        lengths = [len(chunk.content) for chunk in self.document_chunks]
        return {
            'min': min(lengths),
            'max': max(lengths),
            'avg': int(np.mean(lengths))
        }
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Enhanced document retrieval with smart ranking"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            # Use similarity search with relevance scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=min(top_k * 2, len(self.document_chunks))  # Get more for reranking
            )
            
            # Convert to our format and apply enhanced scoring
            results = []
            for doc, score in docs_with_scores:
                # Convert FAISS distance to similarity score (0-1)
                similarity_score = 1 / (1 + score)
                
                # Apply query-specific boosting
                boosted_score = self._apply_query_boosting(query, doc.page_content, similarity_score)
                
                results.append((doc.page_content, boosted_score, doc.metadata))
            
            # Sort by enhanced score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return []
    
    def _apply_query_boosting(self, query: str, content: str, base_score: float) -> float:
        """Apply query-specific boosting to improve relevance"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Keyword matching boost
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        word_overlap = len(query_words & content_words) / len(query_words) if query_words else 0
        
        # Exact phrase boost
        phrase_boost = 0.2 if query_lower in content_lower else 0
        
        # Section relevance boost
        section_boost = 0
        if 'section:' in content_lower:
            for keyword in query_words:
                if keyword in content_lower[:200]:  # Check if keyword is in section title
                    section_boost += 0.1
        
        # Academic content boost
        academic_terms = ['study', 'research', 'analysis', 'method', 'result', 'conclusion']
        academic_boost = sum(0.05 for term in academic_terms if term in content_lower)
        
        # Calculate final score
        enhanced_score = base_score + (word_overlap * 0.3) + phrase_boost + section_boost + min(academic_boost, 0.2)
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def generate_enhanced_answer(self, query: str, context_docs: List[Tuple[str, float, Dict]]) -> str:
        """Generate enhanced answer with proper formatting"""
        if not context_docs:
            return "I couldn't find relevant information in the knowledge base. Please try rephrasing your question or check if the relevant documents are available."
        
        # Filter high-quality results (score > 0.3)
        high_quality_docs = [doc for doc in context_docs if doc[1] > 0.3]
        if not high_quality_docs:
            return "I found some potentially related content, but it doesn't seem directly relevant to your question. Could you try being more specific?"
        
        # Prepare context with smart truncation
        context_parts = []
        total_length = 0
        max_context = 3000  # Increased for research papers
        
        for doc_text, score, metadata in high_quality_docs:
            if total_length + len(doc_text) > max_context:
                remaining = max_context - total_length
                if remaining > 200:
                    context_parts.append(doc_text[:remaining] + "...")
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        combined_context = "\n\n".join(context_parts)
        
        # Extract key information
        answer = self._extract_and_format_answer(query, combined_context, high_quality_docs)
        
        return answer
    
    def _extract_and_format_answer(self, query: str, context: str, docs: List[Tuple[str, float, Dict]]) -> str:
        """Extract and format answer with academic structure"""
        
        # Get sources
        sources = list(set(doc[2].get('source_file', 'Unknown') for doc in docs))
        
        # Identify query type
        query_lower = query.lower()
        is_definition = any(word in query_lower for word in ['what is', 'define', 'explain', 'describe'])
        is_how = any(word in query_lower for word in ['how', 'steps', 'process', 'method'])
        is_why = any(word in query_lower for word in ['why', 'reason', 'cause', 'because'])
        is_comparison = any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus'])
        
        # Extract relevant sentences
        sentences = self._extract_relevant_sentences(query, context)
        
        if not sentences:
            return f"Based on the available documents ({', '.join(sources)}), I couldn't extract specific information to answer your question directly."
        
        # Format answer based on query type
        answer_parts = []
        
        # Introduction
        if is_definition:
            answer_parts.append(f"**Definition/Explanation:**")
        elif is_how:
            answer_parts.append(f"**Process/Method:**")
        elif is_why:
            answer_parts.append(f"**Explanation:**")
        elif is_comparison:
            answer_parts.append(f"**Comparison:**")
        else:
            answer_parts.append(f"**Answer:**")
        
        # Main content
        main_content = self._format_main_content(sentences, is_how)
        answer_parts.append(main_content)
        
        # Add context from sections if available
        section_info = self._extract_section_context(docs)
        if section_info:
            answer_parts.append(f"\n**Context:** {section_info}")
        
        # Sources
        answer_parts.append(f"\n**Sources:** {', '.join(sources)}")
        
        # Confidence indicator
        avg_score = np.mean([doc[1] for doc in docs])
        if avg_score > 0.7:
            confidence = "High"
        elif avg_score > 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        answer_parts.append(f"*Confidence: {confidence}*")
        
        return "\n\n".join(answer_parts)
    
    def _extract_relevant_sentences(self, query: str, context: str) -> List[str]:
        """Extract sentences most relevant to the query"""
        sentences = re.split(r'[.!?]+', context)
        query_words = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            
            # Calculate relevance score
            relevance = overlap / len(query_words) if query_words else 0
            
            # Boost if sentence contains key academic indicators
            if any(indicator in sentence.lower() for indicator in ['research', 'study', 'analysis', 'findings', 'results']):
                relevance += 0.1
            
            if relevance > 0.1:  # Only include sentences with some relevance
                relevant_sentences.append((sentence, relevance, overlap))
        
        # Sort by relevance and return top sentences
        relevant_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [sent[0] for sent in relevant_sentences[:5]]
    
    def _format_main_content(self, sentences: List[str], is_process: bool = False) -> str:
        """Format main content based on query type"""
        if not sentences:
            return "No specific information found."
        
        if is_process and len(sentences) > 1:
            # Format as numbered steps
            formatted = []
            for i, sentence in enumerate(sentences[:4], 1):
                formatted.append(f"{i}. {sentence.strip()}")
            return "\n".join(formatted)
        else:
            # Format as paragraphs
            return " ".join(sentences[:3])
    
    def _extract_section_context(self, docs: List[Tuple[str, float, Dict]]) -> str:
        """Extract section context from documents"""
        sections = []
        for _, _, metadata in docs:
            section = metadata.get('section', '')
            if section and section not in sections:
                sections.append(section)
        
        if sections:
            return f"Information found in sections: {', '.join(sections[:3])}"
        return ""

# Simple demo function - no hardcoded content
def create_empty_documents_folder():
    """Create empty documents folder structure"""
    docs_dir = "data/documents"
    os.makedirs(docs_dir, exist_ok=True)
    
    readme_content = """# Document Repository

This folder contains your documents for the RAG system.

## Supported Formats:
- PDF files (.pdf) - Research papers, reports, etc.
- Text files (.txt) - Plain text documents

## Usage:
1. Add your documents to this folder
2. Run: python rag_system.py
3. Start the chatbot: streamlit run app.py

## Features:
- Advanced PDF processing with image detection
- Smart chunking for research papers
- Section-aware text splitting
- Enhanced embedding generation
"""
    
    with open(os.path.join(docs_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"✅ Created documents directory: {docs_dir}")
    print("📄 Add your PDF and TXT files to this directory")

if __name__ == "__main__":
    # Example usage
    try:
        print("🚀 Starting Enhanced RAG System...")
        rag = EnhancedRAGSystem()
        
        # Check for documents directory
        docs_dir = "data/documents"
        if not os.path.exists(docs_dir) or not any(Path(docs_dir).iterdir()):
            print("📁 Creating documents directory...")
            create_empty_documents_folder()
            print("\n🚨 Please add your PDF and TXT files to data/documents/ directory")
            print("Then run this script again to generate embeddings.")
            exit(0)
        
        # Generate embeddings
        print("⚡ Generating enhanced embeddings...")
        rag.create_embeddings()
        
        print("\n✅ Enhanced RAG system setup complete!")
        print("You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()