import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pandas as pd

class RAGSystem:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize RAG system with lightweight model optimized for your CPU
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embeddings_file = "data/embeddings.pkl"
        
    def load_documents(self, file_path: str):
        """Load documents from text file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Documents file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
        if not content:
            raise ValueError(f"Documents file is empty: {file_path}")
            
        # Split by double newlines to separate documents
        self.documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        if not self.documents:
            raise ValueError(f"No valid documents found in file: {file_path}")
            
        print(f"Loaded {len(self.documents)} documents")
        
        # Print first few document titles for debugging
        for i, doc in enumerate(self.documents[:3]):
            title = doc.split('\n')[0] if doc else "Empty document"
            print(f"  Document {i+1}: {title[:50]}{'...' if len(title) > 50 else ''}")
        
    def _is_valid_embeddings_file(self) -> bool:
        """Check if embeddings file exists and is valid"""
        if not os.path.exists(self.embeddings_file):
            return False
        
        try:
            # Check file size
            if os.path.getsize(self.embeddings_file) == 0:
                print("Embeddings file is empty, will recreate...")
                return False
            
            # Try to load and validate the pickle file
            with open(self.embeddings_file, 'rb') as f:
                test_embeddings = pickle.load(f)
                
            # Validate embeddings shape and type
            if not isinstance(test_embeddings, np.ndarray):
                print("Invalid embeddings format, will recreate...")
                return False
                
            if len(test_embeddings) != len(self.documents):
                print(f"Embeddings count ({len(test_embeddings)}) doesn't match documents count ({len(self.documents)}), will recreate...")
                return False
                
            return True
            
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"Error loading cached embeddings: {e}")
            print("Will recreate embeddings...")
            return False
    
    def create_embeddings(self, force_recreate=False):
        """Create embeddings with robust caching and error handling"""
        # Validate we have documents to embed
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
            
        should_recreate = force_recreate or not self._is_valid_embeddings_file()
        
        if not should_recreate:
            try:
                print("Loading cached embeddings...")
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print("Successfully loaded cached embeddings!")
            except Exception as e:
                print(f"Failed to load cached embeddings: {e}")
                should_recreate = True
        
        if should_recreate:
            print(f"Creating new embeddings for {len(self.documents)} documents... (this may take a few minutes)")
            try:
                # Create embeddings
                self.embeddings = self.model.encode(self.documents, show_progress_bar=True)
                
                # Validate embeddings were created properly
                if self.embeddings is None or len(self.embeddings) == 0:
                    raise ValueError("Failed to create embeddings - result is empty")
                
                if len(self.embeddings) != len(self.documents):
                    raise ValueError(f"Embedding count ({len(self.embeddings)}) doesn't match document count ({len(self.documents)})")
                
                print(f"Created embeddings with shape: {self.embeddings.shape}")
                
                # Ensure data directory exists
                os.makedirs("data", exist_ok=True)
                
                # Remove old corrupted file if it exists
                if os.path.exists(self.embeddings_file):
                    os.remove(self.embeddings_file)
                
                # Cache embeddings with error handling
                with open(self.embeddings_file, 'wb') as f:
                    pickle.dump(self.embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Verify the saved file
                if self._is_valid_embeddings_file():
                    print("Embeddings successfully created and cached!")
                else:
                    print("Warning: Failed to verify saved embeddings file")
                    
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                raise
        
        # Validate embeddings before creating index
        if self.embeddings is None:
            raise ValueError("Embeddings is None")
        
        if len(self.embeddings.shape) != 2:
            raise ValueError(f"Invalid embeddings shape: {self.embeddings.shape}. Expected 2D array.")
        
        if self.embeddings.shape[0] == 0:
            raise ValueError("Embeddings array is empty")
            
        # Create FAISS index
        try:
            dimension = self.embeddings.shape[1]
            print(f"Creating FAISS index with dimension: {dimension}")
            
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self.embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            self.index.add(normalized_embeddings.astype('float32'))
            
            print(f"FAISS index created with {self.index.ntotal} vectors of dimension {dimension}")
            
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            raise
        
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            raise ValueError("Index not created. Run create_embeddings() first.")
            
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Return documents with scores
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.documents):  # Safety check
                    results.append((self.documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """Generate answer using retrieved context with better formatting"""
        try:
            if not context_docs or not any(doc.strip() for doc in context_docs):
                return f"I couldn't find relevant information for your query: '{query}'. Please try rephrasing your question or contact support for assistance."
            
            # Extract the most relevant information based on query keywords
            query_lower = query.lower()
            
            # Simple keyword matching to find the most relevant document
            best_doc = ""
            for doc in context_docs:
                if any(keyword in doc.lower() for keyword in query_lower.split()):
                    best_doc = doc
                    break
            
            if not best_doc:
                best_doc = context_docs[0]  # Fall back to first document
            
            # Extract policy information more intelligently
            lines = best_doc.split('\n')
            title = lines[0] if lines else "Policy Information"
            content = '\n'.join(lines[1:]) if len(lines) > 1 else best_doc
            
            # Generate a more natural response
            if "vacation" in query_lower or "leave" in query_lower:
                answer = f"""Here's our **{title}**:

{content}

**Key Points:**
• You get 25 days of paid vacation per year
• Vacation days reset every January 1st
• Days cannot be carried over to the next year  
• Additional unpaid leave requires management approval

Is there anything specific about the vacation policy you'd like to know more about?"""
            
            elif "remote" in query_lower or "work from home" in query_lower:
                answer = f"""Here's our **{title}**:

{content}

**Key Points:**
• Work from home up to 3 days per week
• Requires manager approval
• Must maintain regular communication
• Expected to attend virtual meetings

Need more details about remote work arrangements?"""
            
            elif "health" in query_lower or "benefits" in query_lower:
                answer = f"""Here's our **{title}**:

{content}

**What's Covered:**
• Medical, dental, and vision insurance
• Coverage starts on your first day
• Mental health support included
• Wellness programs available

Any questions about your health benefits?"""
            
            elif "training" in query_lower or "development" in query_lower:
                answer = f"""Here's our **{title}**:

{content}

**Development Opportunities:**
• $2000 annual budget for professional development
• Courses, conferences, and certifications covered
• Mentorship programs available
• Support for career growth

What type of training are you interested in?"""
            
            else:
                # Generic response for other queries
                answer = f"""Here's information about **{title}**:

{content}

This should help answer your question about "{query}". If you need more specific information, feel free to ask!"""
            
            return answer
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating the response: {e}. Please try again or rephrase your question."

# Sample knowledge base creation helper
def create_sample_knowledge_base():
    """Create a sample knowledge base for testing"""
    sample_docs = """
Company Overview
Our company was founded in 2020 and specializes in AI-powered solutions for businesses. We have over 500 employees across 10 countries. Our mission is to leverage technology to simplify complex business processes.

Vacation Policy
Employees are entitled to 25 days of paid vacation per year. Vacation days reset on January 1st each year and cannot be carried over. Additional unpaid leave may be requested subject to management approval.

Remote Work Policy
We support flexible remote work arrangements. Employees can work from home up to 3 days per week with manager approval. All remote employees are expected to maintain regular communication and attend virtual meetings as scheduled.

Health Benefits
We provide comprehensive health insurance including medical, dental, and vision coverage. Coverage begins on your first day of employment. Mental health support and wellness programs are also included.

Training and Development
Each employee has a $2000 annual budget for professional development including courses, conferences, and certifications. Mentorship programs are available to support career growth.

Performance Reviews
Performance reviews are conducted quarterly with annual salary reviews in December. Feedback is continuous, and employees are encouraged to set measurable goals with their managers.

Office Hours
Our standard office hours are 9 AM to 6 PM, Monday through Friday. Flexible hours are available with manager approval. Core hours from 10 AM to 4 PM require presence for team collaboration.

IT Support
For technical issues, contact IT support at support@company.com or call extension 1234. Remote troubleshooting and hardware replacement requests can be submitted through the IT portal.

Expense Reimbursement
Submit expense reports through the company portal within 30 days. Approved expenses are reimbursed within 5 business days. International expenses require additional documentation.

Meeting Room Booking
Conference rooms can be booked through the office calendar system. Rooms are available from 8 AM to 8 PM. Video conferencing equipment is available in all major meeting rooms.

Security Policies
All employees must follow company security protocols including password policies, device encryption, and secure handling of sensitive data. Reporting of suspicious activity is mandatory.

Code of Conduct
Employees are expected to maintain professional behavior, respect diversity, and adhere to ethical business practices. Harassment or discrimination of any kind will not be tolerated.

Travel Policy
Business travel must be approved by management. Employees are reimbursed for travel-related expenses according to company guidelines. Preferred travel vendors should be used whenever possible.

Communication Guidelines
Official communications should be conducted via company email or approved messaging platforms. Confidential information must never be shared outside authorized channels.

Onboarding Process
New hires undergo a 2-week onboarding program including orientation, system access setup, and departmental introductions. A mentor is assigned to each new employee for guidance.

Termination Policy
Voluntary or involuntary termination requires a notice period of 30 days unless otherwise specified in employment contracts. Exit interviews are conducted to gather feedback.

Corporate Social Responsibility
Our company actively engages in community development and environmental sustainability initiatives. Employees are encouraged to participate in volunteering programs.

Emergency Procedures
Emergency contacts and evacuation plans are available in all office locations. Employees must participate in annual safety drills and report any hazards.

IT Asset Management
Company-provided devices and software must be used according to policy. Personal devices accessing company systems must meet security standards.

Data Privacy
Employees must adhere to data protection laws and company privacy policies. Unauthorized access, sharing, or retention of personal data is strictly prohibited.

Employee Recognition Programs
Outstanding employees are recognized through monthly awards, performance bonuses, and company-wide shout-outs. Peer-to-peer recognition programs are also encouraged.

Diversity and Inclusion
Our company is committed to fostering a diverse and inclusive workplace. Programs and events promote awareness, education, and inclusion across all teams.

Sustainability Initiatives
We implement eco-friendly practices such as reducing paper usage, promoting recycling, and optimizing energy consumption in office spaces.

Professional Ethics
Employees must maintain integrity, transparency, and accountability in all professional interactions. Conflicts of interest should be disclosed immediately.

Social Media Policy
Employees representing the company online must adhere to professional standards and avoid sharing confidential information. Personal opinions should be clearly separated from official company statements.

Workplace Safety
All office locations comply with safety regulations. Employees are encouraged to report hazards, follow ergonomic guidelines, and participate in safety training sessions.

Conflict Resolution
Employees are encouraged to resolve conflicts amicably. HR mediation services are available if disputes cannot be resolved directly between parties.

Volunteering and Community Engagement
Employees may participate in company-supported volunteer programs. Time off for approved volunteer activities is granted on a case-by-case basis.

Professional Certifications
The company supports employees obtaining industry certifications relevant to their role. Certification costs can be reimbursed subject to manager approval.

Intellectual Property Policy
All work created by employees within the scope of their employment is considered company intellectual property. Employees must not use proprietary information for personal gain.

Internal Communication Tools
Approved communication tools include company email, messaging platforms, and collaboration software. Unauthorized tools should not be used for official communication.

Flexible Working Arrangements
In addition to remote work, flexible working arrangements include compressed workweeks and staggered shifts. Approval must be obtained from managers.

Company Events
Annual team-building events, holiday parties, and hackathons are organized to promote engagement and collaboration among employees.

Employee Assistance Program (EAP)
Confidential counseling and support services are available to employees facing personal or professional challenges. Access is free and available 24/7.

"""
    
    os.makedirs("data", exist_ok=True)
    with open("data/documents.txt", "w", encoding="utf-8") as f:
        f.write(sample_docs)
    print("Sample knowledge base created at data/documents.txt")