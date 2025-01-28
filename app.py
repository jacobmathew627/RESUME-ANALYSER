import base64
import io
from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from PIL import Image
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Updated function using the new model "gemini-1.5-flash"
def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Using the new model here
    response = model.generate_content([input, pdf_content[0], prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        poppler_path = r'C:\Program Files\poppler\Library\bin'  # Make sure to set the correct path
        try:
            images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
            first_page = images[0]
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                }
            ]
            return pdf_parts
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {e}")
    else:
        raise FileNotFoundError("File not found")

st.set_page_config(page_title="ats resume expert", page_icon="🔮", layout="wide")
st.header('🔮 ATS ')
input_text = st.text_area('Job Description', key='input_text')
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")
    submit1 = st.button("Extract Text")
    submit2 = st.button("Match With Job Description")
    submit3 = st.button("What are the Keywords That are Missing")
    submit4 = st.button("ATS Compatibility")

    input_prompt1 = '''
    You are an advanced AI system capable of parsing and understanding the structure of resumes. Your task is to extract relevant information from the uploaded resume. Focus on key sections such as:

    Personal Information (Name, Contact Info)
    Objective/Professional Summary
    Skills
    Work Experience (Job Titles, Employers, Dates, Responsibilities)
    Education (Degrees, Institutions, Dates)
    Certifications and Achievements
    Languages and Interests (if applicable)
    You must ensure that the extracted data maintains its context and relationships. For example, the job titles should align with their corresponding companies and dates, and the skills section should be accurately linked to the candidate's experience. You should also preserve any formatting that reflects the resume's structure, such as bullet points for responsibilities or list items for skills.
    '''

    input_prompt2 = '''
    You are an AI system specializing in matching resumes to job descriptions. Your task is to evaluate the relevance of the provided resume against the given job description. Specifically, analyze the following:

    Job Title and Role Requirements: Compare the job title and key responsibilities outlined in the job description with the candidate's work experience. Check if the candidate’s past roles and responsibilities match the job requirements.

    Skills and Competencies: Identify the skills listed in the job description and match them with the skills mentioned in the resume. Assess if the resume contains the required skills or relevant experience.

    Qualifications and Education: Cross-reference the educational qualifications and certifications required in the job description with those listed in the resume.

    Keywords and Phrases: Match any specific keywords or phrases in the job description (e.g., programming languages, tools, or methodologies) with those found in the resume. Identify if important terms are missing or mismatched.

    For each section, provide an overall matching score indicating how closely the resume aligns with the job description. In case of low matching areas, suggest actionable improvements or modifications to make the resume more tailored to the job description, such as adding missing skills, clarifying job roles, or reformatting certain sections for better alignment.
    '''

    input_prompt3 = '''
    You are an AI system tasked with extracting keywords from a resume and identifying any missing keywords based on the provided job description. Follow these steps:

    Extract Keywords from the Resume: Identify the most important keywords in the resume. This includes skills, certifications, job titles, industry-specific terminology, tools, software, programming languages, and any other significant terms related to the candidate’s experience and qualifications.

    Identify Keywords from the Job Description: Extract relevant keywords from the job description, such as required skills, qualifications, tools, programming languages, certifications, and specific responsibilities.

    Compare and Identify Missing Keywords: Compare the keywords extracted from the resume with the job description. Identify any crucial keywords from the job description that are not present in the resume. Provide a list of missing keywords, along with suggestions for how they can be incorporated into the resume to improve its alignment with the job description.

    For each missing keyword, suggest a way to include it naturally based on the candidate’s experience or qualifications. This could involve rephrasing existing information or highlighting relevant skills and experiences more clearly. Your goal is to help the candidate enhance their resume by including the essential keywords that match the job requirements.
    '''

    input_prompt4 = '''
    Evaluate the ATS compatibility of the uploaded resume. Check for the following:

    Formatting: Ensure proper section headings, simple fonts, and the use of standard formatting (e.g., no images, tables, or text boxes).
    Keyword Optimization: Identify whether the resume includes essential keywords from the job description and is free of keyword stuffing.
    File Structure: Confirm the file is in a readable format (DOCX, PDF) and doesn’t contain elements that could hinder ATS parsing.
    Contact Information: Ensure clear, readable contact details in a standard format.
    Provide a score (1-10) for ATS compatibility and list specific improvements to ensure better ATS parsing.
    '''

    if submit1:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt1)
            st.subheader("Extracted Text")
            st.write(response)
        else:
            st.write("Please upload a PDF file")
    elif submit2:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt2)
            st.subheader("Match With Job Description")
            st.write(response)
    elif submit3:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt3)
            st.subheader("Keywords That are Missing")
            st.write(response)
    elif submit4:
        if uploaded_file is not None:
            pdf_content = input_pdf_setup(uploaded_file)
            response = get_gemini_response(input_text, pdf_content, input_prompt4)
            st.subheader("ATS Compatibility")
            st.write(response)
