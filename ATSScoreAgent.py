import json
import google.generativeai as genai
from config import embedding_model

class ATSScoreAgent:
    def calculate_ats_score(self, resume_json, job_description):
        """Calculates ATS compatibility score (1-10) and suggests improvements."""
        prompt = f"""
        You are an ATS scoring expert.
        Compare the **resume** and **job description**.
        Identify **missing skills**, **keyword matches**, and **alignment score**.
        Provide **suggestions** to improve the resume.

        Resume (JSON format):
        {json.dumps(resume_json, indent=2)}

        Job Description:
        {job_description}

        Return structured JSON:
        {{
            "ats_score": number (1-10),
            "missing_skills": [list of strings],
            "keyword_matches": [list of strings],
            "improvement_suggestions": [list of strings],
            "section_scores": {{
                "skills": number (1-10),
                "experience": number (1-10),
                "education": number (1-10),
                "overall_format": number (1-10)
            }},
            "detailed_analysis": "string describing why the resume received this score",
            "keyword_density": {{
                "resume_keyword_count": number,
                "job_description_keyword_count": number,
                "match_percentage": number
            }}
        }}
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt])
            return json.loads(response.text.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            return {
                "ats_score": 5,
                "missing_skills": ["Error extracting skills"],
                "keyword_matches": [],
                "improvement_suggestions": ["Please try again"],
                "section_scores": {
                    "skills": 5,
                    "experience": 5,
                    "education": 5,
                    "overall_format": 5
                },
                "detailed_analysis": f"Error analyzing resume: {str(e)}",
                "keyword_density": {
                    "resume_keyword_count": 0,
                    "job_description_keyword_count": 0,
                    "match_percentage": 0
                },
                "error": str(e)
            }
    
    def compare_before_after(self, original_resume, optimized_resume, job_description):
        """Compares original and optimized resumes to show improvements"""
        prompt = f"""
        You are an ATS scoring expert.
        Compare the **original resume**, **optimized resume** and **job description**.
        Identify how the optimization improved the resume's ATS score.

        Original Resume:
        {original_resume}

        Optimized Resume:
        {optimized_resume}

        Job Description:
        {job_description}

        Return structured JSON:
        {{
            "original_score": number (1-10),
            "optimized_score": number (1-10),
            "score_improvement": number,
            "key_improvements": [list of strings],
            "added_keywords": [list of strings],
            "reformatted_sections": [list of strings],
            "before_after_analysis": "string analyzing the key differences"
        }}
        """
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt])
            return json.loads(response.text.replace("```json", "").replace("```", "").strip())
        except Exception as e:
            return {
                "original_score": 5,
                "optimized_score": 7,
                "score_improvement": 2,
                "key_improvements": ["Error analyzing improvements"],
                "added_keywords": [],
                "reformatted_sections": [],
                "before_after_analysis": f"Error analyzing differences: {str(e)}",
                "error": str(e)
            }