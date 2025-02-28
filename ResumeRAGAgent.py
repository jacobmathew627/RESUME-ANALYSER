import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import GOOGLE_API_KEY, embedding_model

class ResumeRAGAgent:
    def __init__(self):
        self.dimension = 384  # Size of embeddings from all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.document_store = []  # Stores text and metadata
        
    def add_to_index(self, text, metadata=None):
        """Add a document to the vector index with optional metadata."""
        if not text:
            return
            
        embedding = embedding_model.encode(text).astype('float32').reshape(1, -1)
        self.index.add(embedding)
        self.document_store.append({
            "text": text,
            "metadata": metadata or {}
        })
        return len(self.document_store) - 1  # Return index of added document
        
    def retrieve_similar(self, query_text, top_k=3):
        """Retrieve top-k most similar documents to the query."""
        if self.index.ntotal == 0:
            return []
            
        query_embedding = embedding_model.encode(query_text).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.document_store):
                results.append({
                    "text": self.document_store[idx]["text"],
                    "metadata": self.document_store[idx]["metadata"],
                    "score": float(distances[0][i])
                })
                
        return results
    
    def enhance_resume(self, resume_text, job_description):
        """Enhance a resume using RAG with job description and similar documents."""
        # First, try to find similar successful resumes (if available)
        similar_resumes = self.retrieve_similar(job_description, top_k=2)
        
        # Prepare context with similar resumes as examples
        context = ""
        if similar_resumes:
            context = "Here are examples of successful resumes for similar roles:\n\n"
            for i, resume in enumerate(similar_resumes):
                context += f"Example {i+1}:\n{resume['text'][:500]}...\n\n"
        
        # Use Gemini to enhance the resume
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert resume optimization AI that helps candidates optimize their resumes for specific job descriptions.
        
        Job Description:
        {job_description}
        
        {context}
        
        Candidate's Current Resume:
        {resume_text}
        
        Task: Enhance this resume to better align with the job description while maintaining honesty and accuracy.
        - Improve the formatting and structure
        - Highlight relevant skills and experience
        - Use industry-specific keywords from the job description
        - Make bullet points more achievement-oriented
        - Remove irrelevant information
        - Fix any grammar or spelling issues
        
        Return only the enhanced resume without explanations.
        """
        
        response = model.generate_content([prompt])
        enhanced_resume = response.text
        
        # Create an explanation of changes separately
        explanation_prompt = f"""
        You previously optimized this resume for a job. Explain the key changes you made and why they improve the candidate's chances.
        
        Original Resume:
        {resume_text}
        
        Enhanced Resume:
        {enhanced_resume}
        
        Job Description:
        {job_description}
        
        Provide a concise explanation of:
        1. The main improvements made
        2. How these changes better align with the job requirements
        3. Any key keywords that were added
        4. Which aspects of the resume were strengthened
        """
        
        explanation_response = model.generate_content([explanation_prompt])
        
        return {
            "enhanced_resume": enhanced_resume,
            "explanation": explanation_response.text,
            "similar_resumes_count": len(similar_resumes)
        }
        
    def seed_with_sample_data(self, job_titles=None):
        """Seed the RAG database with some initial example resumes."""
        if job_titles is None:
            job_titles = ["Software Engineer", "Data Scientist", "Project Manager"]
            
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        for title in job_titles:
            # Generate a sample job description
            job_prompt = f"Write a realistic job description for a {title} position."
            job_response = model.generate_content([job_prompt])
            job_description = job_response.text
            
            # Generate a sample good resume
            resume_prompt = f"Write a strong resume for a {title} that would match well with this job description:\n\n{job_description}"
            resume_response = model.generate_content([resume_prompt])
            resume_text = resume_response.text
            
            # Add to index
            self.add_to_index(
                resume_text, 
                metadata={
                    "type": "resume", 
                    "job_title": title,
                    "quality": "high",
                    "matching_job": job_description
                }
            )
            
            # Add job description to index
            self.add_to_index(
                job_description,
                metadata={
                    "type": "job_description",
                    "job_title": title
                }
            )
            
        return len(self.document_store)