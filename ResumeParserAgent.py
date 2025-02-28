import io
import json
import PyPDF2
import docx
import google.generativeai as genai
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from config import embedding_model

class ResumeParserAgent:
    def __init__(self):
        self.keyword_extractor = CountVectorizer(stop_words='english', max_features=100)

    def parse_resume(self, file_bytes, filename):
        """Extracts text, skills, and structured JSON from a resume file."""
        resume_text = self._extract_text(file_bytes, filename)
        skills = self._extract_skills(resume_text)
        resume_json = self._convert_to_json(resume_text)

        return resume_text, skills, resume_json

    def _extract_text(self, file_bytes, filename):
        """Extracts text from PDFs, DOCX, or TXT files."""
        text = ""
        try:
            if filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            elif filename.endswith(".docx"):
                doc = docx.Document(io.BytesIO(file_bytes))
                text = "\n".join([para.text for para in doc.paragraphs])
            elif filename.endswith(".txt"):
                text = file_bytes.decode("utf-8")
            return text
        except Exception as e:
            return f"Error extracting text: {e}"

    def _extract_skills(self, resume_text):
        """Extracts skills using Google Gemini AI."""
        prompt = f"Extract a list of skills from the following resume:\n\n{resume_text[:5000]}"
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt])
            skills = re.split(r',|\n', response.text.strip())
            return [skill.strip() for skill in skills if skill.strip()]
        except Exception:
            return []

    def _convert_to_json(self, resume_text):
        """Converts resume text into structured JSON format."""
        prompt = f"Convert this resume into structured JSON:\n\n{resume_text[:5000]}"
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt])
            return json.loads(response.text.strip("```json").strip("```"))
        except Exception:
            return {"summary": resume_text[:200], "skills": [], "experience": [], "education": []}
