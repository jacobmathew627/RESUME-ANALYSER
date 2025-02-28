import os
import requests
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from config import SERPER_API_KEY, SERPAPI_KEY, GOOGLE_API_KEY

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Size of embeddings from all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
document_store = []  # Stores actual text of resumes/job descriptions

# Configure Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)

# 1️⃣ Job Search Agent
class JobSearchAgent:
    def search_jobs(self, search_query, location=""):
        """Searches for job postings using Google Serper API."""
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY}

        payload = {"q": f"{search_query} jobs in {location}", "num": 5}
        try:
            response = requests.post(url, json=payload, headers=headers)
            response_data = response.json()

            job_results = []
            if "organic" in response_data:
                for result in response_data["organic"]:
                    job_results.append({
                        "title": result.get("title", "Job Posting"),
                        "description": result.get("snippet", "No details available"),
                        "link": result.get("link", "#")
                    })
            return job_results  # ✅ Now returns a list of dictionaries

        except Exception as e:
            print(f"Error retrieving job descriptions: {e}")
            return []

# 2️⃣ Resume Search Agent
class ResumeSearchAgent:
    def search_resumes(self, skill):
        url = "https://serpapi.com/search"
        params = {
            "api_key": SERPAPI_KEY,
            "engine": "google",
            "q": f"resume site:github.com {skill}",
            "num": 5
        }
        response = requests.get(url, params=params)
        resumes = [result["snippet"] for result in response.json()["organic"]]

        # Store resumes in FAISS
        for resume in resumes:
            embedding = model.encode(resume).astype('float32').reshape(1, -1)
            index.add(embedding)
            document_store.append({"type": "resume", "text": resume})

        return resumes

# 3️⃣ Resume Retrieval Agent
class ResumeRetrievalAgent:
    def retrieve_top_resumes(self, job_description, top_k=3):
        query_embedding = model.encode(job_description).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        
        # Filter only resume documents
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(document_store):
                doc = document_store[idx]
                if doc["type"] == "resume":
                    retrieved_docs.append(doc["text"])
                    
        # If we don't have enough resumes, just return what we have
        return retrieved_docs[:top_k]

# 4️⃣ Resume Optimization Agent
class ResumeOptimizationAgent:
    def optimize_resume(self, resume_text, job_description):
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an AI that optimizes resumes for job descriptions.
        Job Description: {job_description}
        
        Candidate Resume:
        {resume_text}
        
        Improve the resume by aligning it with the job description.
        """
        response = model.generate_content([prompt])
        return response.text