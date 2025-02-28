import streamlit as st
import io
import base64
import pdf2image
from PIL import Image
from crew_backend import JobSearchAgent, ResumeSearchAgent, ResumeRetrievalAgent, ResumeOptimizationAgent
from ResumeParserAgent import ResumeParserAgent
from ATSScoreAgent import ATSScoreAgent
from ResumeRAGAgent import ResumeRAGAgent

# Set page config
st.set_page_config(page_title="Resume Optimizer", layout="wide")

# Apply custom styles
st.markdown("""
    <style>
        .results-section {
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            color: #000; 
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .keyword-tag {
            background-color: #e1f5fe;
            padding: 3px 8px;
            border-radius: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
            font-size: 0.8em;
        }
        .missing-tag {
            background-color: #ffebee;
            padding: 3px 8px;
            border-radius: 12px;
            margin-right: 5px;
            margin-bottom: 5px;
            display: inline-block;
            font-size: 0.8em;
        }
        .section-header {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ["resume_text", "extracted_skills", "job_descriptions", "optimized_resume", 
           "ats_results", "resume_json", "rag_results", "before_after_comparison"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key != "extracted_skills" and key != "ats_results" else []

# Initialize RAG agent and seed with sample data if first run
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = ResumeRAGAgent()
    # Seed with initial data in background
    with st.spinner("Initializing AI engine..."):
        st.session_state.rag_agent.seed_with_sample_data()

# Title and description
st.title("🔍 AI-Powered Resume Optimization with RAG")
st.markdown("""
This app uses Retrieval-Augmented Generation (RAG) to help you optimize your resume for specific job descriptions.
Upload your resume, find matching jobs, and get personalized optimization suggestions.
""")

# Create tabs for different app sections
tab1, tab2, tab3 = st.tabs(["📄 Resume Analysis", "💼 Job Matching", "✨ Optimization"])

# Function to process resume
def process_resume(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    parser_agent = ResumeParserAgent()
    resume_text, skills, resume_json = parser_agent.parse_resume(bytes_data, uploaded_file.name)
    return resume_text, skills, resume_json

# Tab 1: Resume Analysis
with tab1:
    st.markdown("### 📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Choose your resume file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            try:
                images = pdf2image.convert_from_bytes(uploaded_file.getvalue())
                st.image(images[0], width=300, caption="Resume Preview")
            except Exception as e:
                st.warning(f"Could not display PDF preview: {e}")

        if st.button("Extract Resume Information"):
            with st.spinner("Processing your resume..."):
                resume_text, skills, resume_json = process_resume(uploaded_file)
                st.session_state.resume_text = resume_text
                st.session_state.extracted_skills = skills
                st.session_state.resume_json = resume_json
                
                # Add resume to RAG index
                st.session_state.rag_agent.add_to_index(
                    resume_text, 
                    metadata={"type": "user_resume", "skills": skills}
                )
                
                st.success("Resume processed successfully!")

    # Display extracted resume information
    if st.session_state.resume_text:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Extracted Resume Content:")
            st.text_area("Resume Text", st.session_state.resume_text[:1000], height=300)
            
        with col2:
            st.markdown("### 🔑 Extracted Skills:")
            # Display skills as tags
            html_skills = ""
            for skill in st.session_state.extracted_skills:
                html_skills += f'<div class="keyword-tag">{skill}</div>'
            st.markdown(html_skills, unsafe_allow_html=True)
            
            # Display basic stats
            st.markdown("### 📊 Resume Statistics:")
            word_count = len(st.session_state.resume_text.split())
            sentence_count = len(st.session_state.resume_text.split('.'))
            st.metric("Word Count", word_count)
            st.metric("Skill Count", len(st.session_state.extracted_skills))
            
            # Check resume format issues
            format_issues = []
            if word_count < 300:
                format_issues.append("Resume appears too short")
            if word_count > 1000:
                format_issues.append("Resume may be too long")
            if len(st.session_state.extracted_skills) < 5:
                format_issues.append("Consider adding more skills")
                
            if format_issues:
                st.markdown("### ⚠️ Format Issues:")
                for issue in format_issues:
                    st.warning(issue)

# Tab 2: Job Matching
with tab2:
    if st.session_state.resume_text:
        st.markdown("### 💼 Find Matching Jobs")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            job_title = st.text_input("Job Title:", "")
        with col2:
            location = st.text_input("Location (optional):", "")

        if st.button("Find Matching Jobs"):
            with st.spinner("Searching for relevant job descriptions..."):
                job_agent = JobSearchAgent()
                search_query = job_title if job_title else " ".join(st.session_state.extracted_skills[:3])
                job_descriptions = job_agent.search_jobs(search_query, location)
                
                # Add job descriptions to RAG index
                for job in job_descriptions:
                    st.session_state.rag_agent.add_to_index(
                        job["description"],
                        metadata={"type": "job_description", "title": job["title"]}
                    )
                
                st.session_state.job_descriptions = job_descriptions
                st.success(f"Found {len(job_descriptions)} relevant job postings!")

        # Display matching job descriptions
        if st.session_state.job_descriptions:
            st.markdown("### 📄 Matching Job Descriptions:")
            
            for i, jd in enumerate(st.session_state.job_descriptions):
                job_title = jd.get("title", f"Job #{i+1}")
                job_description = jd.get("description", "No description available")
                job_link = jd.get("link", "#")
                
                with st.expander(f"🔹 {job_title}"):
                    st.markdown(f"**[Open Job Posting]({job_link})**")
                    st.text_area(f"Description", job_description, height=150)
                    
                    if st.button(f"Select This Job #{i+1}", key=f"select_job_{i}"):
                        st.session_state.selected_job_index = i
                        st.session_state.selected_job = job_description
                        st.success(f"Selected: {job_title}")
                
# Tab 3: Optimization
with tab3:
    if not st.session_state.resume_text:
        st.info("Please upload and process your resume first")
    elif "job_descriptions" not in st.session_state or not st.session_state.job_descriptions:
        st.info("Please find matching jobs first")
    else:
        st.markdown("### ✨ Resume Optimization & Scoring")
        
        if "selected_job_index" not in st.session_state:
            selected_job_index = st.selectbox(
                "Select a job description to optimize your resume for:",
                range(len(st.session_state.job_descriptions)),
                format_func=lambda i: st.session_state.job_descriptions[i]["title"]
            )
            selected_job = st.session_state.job_descriptions[selected_job_index]["description"]
        else:
            selected_job_index = st.session_state.selected_job_index
            selected_job = st.session_state.selected_job
            st.write(f"Optimizing for: **{st.session_state.job_descriptions[selected_job_index]['title']}**")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            standard_optimize = st.button("Standard Optimization (Fast)")
        with col2:
            rag_optimize = st.button("RAG-Enhanced Optimization (Comprehensive)")
        
        if standard_optimize or rag_optimize:
            with st.spinner("Analyzing ATS score and optimizing resume..."):
                # Calculate ATS score
                ats_agent = ATSScoreAgent()
                ats_results = ats_agent.calculate_ats_score(st.session_state.resume_json, selected_job)
                st.session_state.ats_results = ats_results
                
                # Process optimization
                if rag_optimize:
                    # Use RAG-enhanced optimization
                    rag_results = st.session_state.rag_agent.enhance_resume(
                        st.session_state.resume_text, 
                        selected_job
                    )
                    st.session_state.optimized_resume = rag_results["enhanced_resume"]
                    st.session_state.rag_results = rag_results
                else:
                    # Use standard optimization
                    optimization_agent = ResumeOptimizationAgent()
                    optimized_resume = optimization_agent.optimize_resume(st.session_state.resume_text, selected_job)
                    st.session_state.optimized_resume = optimized_resume
                
                # Compare before and after
                comparison = ats_agent.compare_before_after(
                    st.session_state.resume_text,
                    st.session_state.optimized_resume,
                    selected_job
                )
                st.session_state.before_after_comparison = comparison
                
                st.success("Resume optimization complete!")
        
        # Display ATS Score Analysis
        if st.session_state.ats_results:
            st.markdown("<div class='section-header'><h3>📊 ATS Score Analysis</h3></div>", unsafe_allow_html=True)
            ats_data = st.session_state.ats_results
            
            # Score display
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric("⭐ ATS Score", f"{ats_data['ats_score']} / 10")
            with col2:
                match_percentage = round((ats_data['ats_score'] / 10) * 100, 2)
                st.metric("🔍 Resume Match", f"{match_percentage}%")
            with col3:
                if "section_scores" in ats_data:
                    avg_section = sum(ats_data["section_scores"].values()) / len(ats_data["section_scores"])
                    st.metric("📋 Section Average", f"{avg_section:.1f} / 10")
            
            # Section scores if available
            if "section_scores" in ats_data:
                st.markdown("#### Section Scores")
                section_scores = ats_data["section_scores"]
                cols = st.columns(len(section_scores))
                for i, (section, score) in enumerate(section_scores.items()):
                    with cols[i]:
                        st.metric(section.capitalize(), f"{score}/10")
                        st.progress(score/10)
            
            # Display keywords and missing skills
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### ❌ Missing Skills")
                missing_skills_html = ""
                for skill in ats_data.get("missing_skills", []):
                    missing_skills_html += f'<div class="missing-tag">{skill}</div>'
                st.markdown(missing_skills_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### ✅ Keyword Matches")
                keyword_matches_html = ""
                for keyword in ats_data.get("keyword_matches", []):
                    keyword_matches_html += f'<div class="keyword-tag">{keyword}</div>'
                st.markdown(keyword_matches_html, unsafe_allow_html=True)
            
            # Improvement suggestions
            st.markdown("#### 📌 Improvement Suggestions")
            for suggestion in ats_data.get("improvement_suggestions", []):
                st.info(suggestion)
                
            # Detailed analysis if available
            if "detailed_analysis" in ats_data:
                with st.expander("View Detailed Analysis"):
                    st.write(ats_data["detailed_analysis"])
        
        # Display Before/After Comparison
        if "before_after_comparison" in st.session_state and st.session_state.before_after_comparison:
            st.markdown("<div class='section-header'><h3>🔄 Before/After Comparison</h3></div>", unsafe_allow_html=True)
            comparison = st.session_state.before_after_comparison
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.metric("Original Score", f"{comparison['original_score']}/10")
            with col2:
                st.metric("Optimized Score", f"{comparison['optimized_score']}/10")
            with col3:
                st.metric("Improvement", f"+{comparison['score_improvement']}", 
                          delta=comparison['score_improvement'])
            
            # Key improvements
            st.markdown("#### Key Improvements")
            for improvement in comparison.get("key_improvements", []):
                st.success(improvement)
            
            # Added keywords
            if "added_keywords" in comparison and comparison["added_keywords"]:
                st.markdown("#### Added Keywords")
                keywords_html = ""
                for keyword in comparison["added_keywords"]:
                    keywords_html += f'<div class="keyword-tag">{keyword}</div>'
                st.markdown(keywords_html, unsafe_allow_html=True)
            
            # Before/after analysis
            if "before_after_analysis" in comparison:
                with st.expander("Detailed Analysis of Changes"):
                    st.write(comparison["before_after_analysis"])
        
        # Display RAG-specific results if available
        if "rag_results" in st.session_state and st.session_state.rag_results:
            rag_results = st.session_state.rag_results
            if "explanation" in rag_results:
                st.markdown("<div class='section-header'><h3>🧠 RAG Enhancement Insights</h3></div>", unsafe_allow_html=True)
                st.markdown(rag_results["explanation"])
                
                if rag_results.get("similar_resumes_count", 0) > 0:
                    st.info(f"Used {rag_results['similar_resumes_count']} similar resumes as references for optimization")
        
        # Display Optimized Resume
        if st.session_state.optimized_resume:
            st.markdown("<div class='section-header'><h3>✅ Optimized Resume</h3></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area("Optimized Resume", st.session_state.optimized_resume, height=300)
                
                # Download Button for Optimized Resume
                st.download_button(
                    label="Download Optimized Resume",
                    data=st.session_state.optimized_resume,
                    file_name="optimized_resume.txt",
                    mime="text/plain"
                )
            
            with col2:
                st.markdown("#### What to do next:")
                st.markdown("""
                1. Review the optimized resume
                2. Make any additional personal adjustments
                3. Update formatting in your preferred editor
                4. Download and use for your job application
                """)