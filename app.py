# Step 1: Install all necessary libraries
# You will need to run this in your terminal before starting the app:
# pip install streamlit pandas spacy scikit-learn pypdf2 python-docx
# python -m spacy download en_core_web_lg

import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2
import docx
import time
import io

# --- INITIAL SETUP AND PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MRIN Resume Ranker",
    page_icon="🏅",
    layout="wide"
)

# --- LOAD NLP MODEL (runs only once) ---
@st.cache_resource
def load_spacy_model():
    """Load the spaCy NLP model."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("SpaCy model not found. Please run 'python -m spacy download en_core_web_lg'")
        return None

nlp = load_spacy_model()

# --- INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_role' not in st.session_state: st.session_state['user_role'] = None
if 'page' not in st.session_state: st.session_state['page'] = 'Login'
if 'job_description' not in st.session_state: st.session_state['job_description'] = ""
if 'jd_skills' not in st.session_state: st.session_state['jd_skills'] = []
if 'jd_experience' not in st.session_state: st.session_state['jd_experience'] = 3 # Default value
if 'uploaded_resumes' not in st.session_state: st.session_state['uploaded_resumes'] = []
if 'ranking_results' not in st.session_state: st.session_state['ranking_results'] = None

# --- FILE PROCESSING HELPER FUNCTIONS ---
def extract_text_from_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def extract_text_from_docx(file_bytes):
    try:
        document = docx.Document(file_bytes)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return None

# --- NEW: CORE MODEL/RANKING FUNCTIONS ---
def extract_years_of_experience(text):
    pattern = r"(\d+\.?\d*)\s+years?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match: return float(match.group(1))
    return 0

def calculate_skills_score(resume_skills_text, required_skills):
    if not isinstance(resume_skills_text, str) or not resume_skills_text: return 0, []
    resume_skills_set = set([skill.strip().lower() for skill in resume_skills_text.split(',')])
    matched_skills = resume_skills_set.intersection(set(required_skills))
    score = len(matched_skills) / len(required_skills) if required_skills else 0
    return score, list(matched_skills)

def calculate_experience_score(years, required_years):
    if years >= required_years: return 1.0
    elif years > 0: return years / required_years
    return 0.0

def calculate_semantic_similarity(resume_text, jd_text):
    if not nlp: return 0.0
    resume_doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    if resume_doc.vector_norm and jd_doc.vector_norm:
        return resume_doc.similarity(jd_doc)
    return 0.0

def generate_feedback(scores):
    feedback_parts = []
    if scores['skills_score'] >= 0.8: feedback_parts.append(f"✅ Strong skills alignment: Matched {len(scores['matched_skills'])} key skills.")
    elif scores['skills_score'] >= 0.5: feedback_parts.append(f"🟡 Good skills foundation: Matched skills include {', '.join(scores['matched_skills'][:3])}.")
    else: feedback_parts.append(f"❌ Skills Gap: Lacks several key skills required for the role.")

    if scores['experience_score'] == 1.0: feedback_parts.append(f"✅ Excellent Experience: Meets the {scores['required_years']}+ years requirement with {scores['years_experience']} years.")
    elif scores['experience_score'] > 0: feedback_parts.append(f"🟡 Developing Experience: Has {scores['years_experience']} years, which is below the desired {scores['required_years']}+ years.")
    else: feedback_parts.append(f"❌ Lacks Required Experience: Does not meet the {scores['required_years']}+ year requirement.")
    
    if scores['similarity_score'] >= 0.90: feedback_parts.append("✅ High Relevance: Resume content is highly aligned with the job responsibilities.")
    elif scores['similarity_score'] >= 0.80: feedback_parts.append("🟡 Moderate Relevance: Resume content shows good alignment with the role.")
    else: feedback_parts.append("❌ Low Relevance: Resume focus may not align with the core duties.")
    return "\n".join(feedback_parts)

# --- HELPER FUNCTIONS FOR PAGE NAVIGATION ---
def navigate_to(page_name): st.session_state.page = page_name

def logout():
    st.session_state.clear() # Clears all session data
    st.session_state['page'] = 'Login'
    st.rerun()

# --- UI PAGES ---
def show_login_page():
    st.title("🏅 MRIN Resume Ranking System")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Login")
            email = st.text_input("Email / Username")
            password = st.text_input("Password", type="password")
            role = st.radio("Select your role:", ('Recruiter', 'Candidate'), key='login_role')
            if st.button("Login", use_container_width=True):
                if email and password:
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = role
                    st.session_state['page'] = 'Dashboard'
                    st.rerun()
                else: st.error("Please enter both email and password.")

def show_dashboard():
    st.sidebar.title(f"Welcome, {st.session_state['user_role']}!")
    st.sidebar.button("Logout", on_click=logout, use_container_width=True)
    st.sidebar.markdown("---")
    if st.session_state['user_role'] == 'Recruiter': show_recruiter_dashboard()
    else: show_candidate_dashboard()

def show_recruiter_dashboard():
    st.sidebar.header("Navigation")
    if st.sidebar.button("Dashboard Home", use_container_width=True): navigate_to('Dashboard')
    if st.sidebar.button("Upload Job Description", use_container_width=True): navigate_to('Upload JD')
    if st.sidebar.button("Upload Resumes", use_container_width=True): navigate_to('Upload Resumes')
    if st.sidebar.button("View Ranked Results", use_container_width=True): navigate_to('View Results')

    if st.session_state.page == 'Dashboard':
        st.header("Recruiter Dashboard")
        st.info(f"**Current Status:**\n\n"
                f"- Job Description: {'Uploaded' if st.session_state.job_description else 'Not Uploaded'}\n\n"
                f"- Resumes: {len(st.session_state.uploaded_resumes)} uploaded")
    elif st.session_state.page == 'Upload JD': show_jd_upload_page()
    elif st.session_state.page == 'Upload Resumes': show_resume_upload_page()
    elif st.session_state.page == 'View Results': show_ranked_results_page()

def show_jd_upload_page():
    st.header("📝 Upload Job Description")
    jd_text_area = st.text_area("Paste Job Description here:", height=300, value=st.session_state.job_description)
    jd_file = st.file_uploader("Or upload a JD file:", type=['pdf', 'docx'])
    
    # NEW: Add fields for extracting key JD info
    st.subheader("Extract Key Information")
    skills_text = st.text_input("Enter required skills (comma-separated):", "Python, SQL, Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Tableau, R")
    experience_years = st.number_input("Enter required years of experience:", min_value=0, max_value=30, value=5)

    if st.button("Save Job Description", use_container_width=True):
        final_jd_text = jd_text_area
        if jd_file:
            final_jd_text += "\n" + (extract_text_from_pdf(io.BytesIO(jd_file.getvalue())) if jd_file.type == "application/pdf" else extract_text_from_docx(io.BytesIO(jd_file.getvalue())))
        
        if final_jd_text and skills_text:
            st.session_state.job_description = final_jd_text
            st.session_state.jd_skills = [skill.strip().lower() for skill in skills_text.split(',')]
            st.session_state.jd_experience = experience_years
            st.success("Job Description saved successfully!")
            time.sleep(1); navigate_to('Dashboard'); st.rerun()
        else: st.error("Please provide both the JD content and the required skills.")

def show_resume_upload_page():
    st.header("📄 Upload Resumes")
    uploaded_files = st.file_uploader("Drag & Drop Resumes (PDF or DOCX):", accept_multiple_files=True, key="resume_uploader")

    if st.button("Process Uploaded Resumes", use_container_width=True, disabled=not uploaded_files):
        st.session_state.uploaded_resumes = []
        progress_bar = st.progress(0, "Starting processing...")
        for i, file in enumerate(uploaded_files):
            content = (extract_text_from_pdf(io.BytesIO(file.getvalue())) if file.type == "application/pdf" else extract_text_from_docx(io.BytesIO(file.getvalue())))
            if content: st.session_state.uploaded_resumes.append({"name": file.name, "content": content})
            progress_bar.progress((i + 1) / len(uploaded_files), f"Processing {file.name}...")
        progress_bar.success("All resumes processed!"); time.sleep(1); navigate_to('Dashboard'); st.rerun()

    if st.session_state.uploaded_resumes:
        st.subheader(f"{len(st.session_state.uploaded_resumes)} Resumes Stored:")
        for resume in st.session_state.uploaded_resumes: st.write(f"📄 {resume['name']}")

def show_ranked_results_page():
    st.header("🏆 Ranked Candidate Results")
    if not st.session_state.job_description or not st.session_state.uploaded_resumes:
        st.warning("Please make sure you have uploaded both a Job Description and Resumes before ranking.")
        col1, col2 = st.columns(2)
        with col1: st.button("Go to Upload JD", on_click=navigate_to, args=('Upload JD',), use_container_width=True)
        with col2: st.button("Go to Upload Resumes", on_click=navigate_to, args=('Upload Resumes',), use_container_width=True)
        return

    if st.button("Start Ranking Process", use_container_width=True, type="primary"):
        with st.spinner("Analyzing resumes... This may take a moment."):
            results = []
            weights = {'skills': 0.50, 'experience': 0.30, 'similarity': 0.20} # Adjustable weights
            
            for resume in st.session_state.uploaded_resumes:
                resume_text = resume['content']
                # Extract skills from the full resume text for more accuracy
                # A simple regex for skills section, can be improved
                skills_match = re.search(r"skills\s*:\s*(.*)", resume_text, re.IGNORECASE)
                resume_skills_text = skills_match.group(1) if skills_match else resume_text

                skills_score, matched_skills = calculate_skills_score(resume_skills_text, st.session_state.jd_skills)
                years_experience = extract_years_of_experience(resume_text)
                experience_score = calculate_experience_score(years_experience, st.session_state.jd_experience)
                similarity_score = calculate_semantic_similarity(resume_text, st.session_state.job_description)
                
                final_score = (weights['skills'] * skills_score + weights['experience'] * experience_score + weights['similarity'] * similarity_score) * 100
                
                score_details = {'skills_score': skills_score, 'experience_score': experience_score, 'similarity_score': similarity_score,
                                 'years_experience': years_experience, 'required_years': st.session_state.jd_experience, 'matched_skills': matched_skills}
                
                results.append({'Candidate': resume['name'], 'Score': final_score, 'Skills Match': f"{skills_score*100:.1f}%",
                                'Experience Match': f"{experience_score*100:.1f}%", 'Feedback': generate_feedback(score_details)})
            
            st.session_state.ranking_results = pd.DataFrame(results).sort_values(by='Score', ascending=False).reset_index(drop=True)

    if st.session_state.ranking_results is not None:
        st.success("Ranking complete!")
        st.dataframe(st.session_state.ranking_results[['Candidate', 'Score', 'Skills Match', 'Experience Match']], use_container_width=True)
        
        st.subheader("Detailed Feedback")
        for index, row in st.session_state.ranking_results.iterrows():
            with st.expander(f"**{index + 1}. {row['Candidate']}** (Score: {row['Score']:.2f})"):
                st.write(row['Feedback'])

# --- MAIN APP ROUTER ---
if not st.session_state.logged_in:
    show_login_page()
else:
    show_dashboard()

