import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API Keys
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Vector Storage
dimension = 384  # Size of embeddings
index = faiss.IndexFlatL2(dimension)
document_store = []  # Stores resumes/job descriptions with indices

# 1️⃣ Job Search Agent (Collects job descriptions)
class JobSearchAgent(Agent):
    def search_jobs(self, job_title, location):
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY}
        payload = {"q": f"{job_title} jobs in {location}", "num": 5}
        response = requests.post(url, json=payload, headers=headers)
        job_descriptions = [result["snippet"] for result in response.json()["organic"]]
        
        # Store job descriptions in FAISS
        for job in job_descriptions:
            embedding = model.encode([job])
            index.add(np.array(embedding))
            document_store.append(job)
        
        return job_descriptions

# 2️⃣ Resume Search Agent (Finds resumes on GitHub)
class ResumeSearchAgent(Agent):
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
            embedding = model.encode([resume])
            index.add(np.array(embedding))
            document_store.append(resume)
        
        return resumes

# 3️⃣ Resume Retrieval Agent (Finds best resumes for a job)
class ResumeRetrievalAgent(Agent):
    def retrieve_top_resumes(self, job_description, top_k=3):
        query_embedding = model.encode([job_description])
        distances, indices = index.search(np.array(query_embedding), top_k)
        retrieved_resumes = [document_store[idx] for idx in indices[0]]
        return retrieved_resumes

# 4️⃣ Resume Optimization Agent (Uses Gemini to enhance resumes)
class ResumeOptimizationAgent(Agent):
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

# Instantiate Agents
job_agent = JobSearchAgent()
resume_agent = ResumeSearchAgent()
retrieval_agent = ResumeRetrievalAgent()
optimization_agent = ResumeOptimizationAgent()

# Crew Task: Collect job descriptions & resumes
class DataCollectionTask(Task):
    def __init__(self, job_title, location, skill):
        self.job_title = job_title
        self.location = location
        self.skill = skill

    def run(self):
        job_descriptions = job_agent.search_jobs(self.job_title, self.location)
        resumes = resume_agent.search_resumes(self.skill)
        return f"Collected {len(job_descriptions)} job descriptions and {len(resumes)} resumes."

# Crew Task: Retrieve & Optimize Resumes
class ResumeMatchingTask(Task):
    def __init__(self, job_description):
        self.job_description = job_description

    def run(self):
        relevant_resumes = retrieval_agent.retrieve_top_resumes(self.job_description, top_k=3)
        optimized_resumes = [optimization_agent.optimize_resume(res, self.job_description) for res in relevant_resumes]
        return optimized_resumes

# Create Crew
crew = Crew(
    agents=[job_agent, resume_agent, retrieval_agent, optimization_agent],
    tasks=[
        DataCollectionTask("Software Engineer", "New York", "Python Developer"),
        ResumeMatchingTask("Looking for a Python Developer with AI experience")
    ]
)

# Run CrewAI Workflow
crew.kickoff()
