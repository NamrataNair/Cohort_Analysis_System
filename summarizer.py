# summarizer.py

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from langchain_ollama.llms import OllamaLLM
import re
import json

# Function to clean the value of a cell
# - Remove HTML tags
def clean_value(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unknown"
    text = BeautifulSoup(str(val), "html.parser").get_text(separator="\n")
    return re.sub(r'\s+', ' ', text).strip()

# combine all the columns into a single string
def format_row(row):
    row_dict = row.to_dict()
    formatted = ""
    for col, val in row_dict.items():
        formatted += f"{col}: {clean_value(val)}\n"
    return formatted.strip()

# Remove markdown formatting from the text returned by the LLM
def clean_markdown_from_llm(text):
    # Remove bold and italic markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove markdown headings and bullet symbols
    text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
    
    # collapse multiple newlines
    text = re.sub(r'\n{2,}', '\n\n', text)
    
    return text.strip()

# Function to ensure the ICD-10 code has exactly one digit after the decimal point
# - 'C18'     → 'C18.0'
def ensure_one_decimal(icd_code):
    """
    Ensures the ICD-10 code has exactly one digit after the decimal point.
    - 'C18'     → 'C18.0'
    - 'C18.9'   → 'C18.9'
    - 'Z51.91'  → 'Z51.9'
    - 'C78.00'  → 'C78.0'
    """
    if '.' in icd_code:
        prefix, suffix = icd_code.split('.')
        first_decimal = suffix[0] if suffix else '0'
        return f"{prefix}.{first_decimal}"
    else:
        return f"{icd_code}.0"

# Function to map ICD codes with their names
# - Load the mapping from icd_codes_with_desc.json
def map_icd_codes_with_names(icd_codes):
    # map icd_ocdes with the name from icd_codes_with_desc.json
    try:
        with open("icd_codes_with_desc.json", "r") as f:
            icd_codes_with_desc = json.load(f)
    except FileNotFoundError:
        return icd_codes
    except json.JSONDecodeError:
        return icd_codes

    # parse the icd_codes json string 
    try:
        icd_codes = json.loads(icd_codes)
    except json.JSONDecodeError:
        return icd_codes

    # map the icd codes with the names
    cleaned_icd_codes = []
    for icd_code in icd_codes:
        code = icd_code.get("code")
        code = ensure_one_decimal(code)
        if code in icd_codes_with_desc:
            cleaned_icd_codes.append({
                "code": code,
                "name": icd_codes_with_desc[code]
            })
        else:
            cleaned_icd_codes.append({
                "code": code,
                "name": "Unknown"
            })
    return cleaned_icd_codes
    


# Function to summarize the text with the LLM
def summarize_with_llm(text, prompt, model):
    prompt = (f'''
    {prompt}
    
    [DATA]
    {text}
    ''')
    try:

        print(f"Prompt: {prompt}")
        
        res = model.invoke(prompt)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()

        cleaned_res = clean_markdown_from_llm(cleaned_res)

        print(f"LLM Response: {cleaned_res}")
        return cleaned_res
    except Exception as e:
        return f"LLM Error: {e}"

# Function to extract ICD codes with the LLM

# Change the model to an appropriate one for ICD code extraction
def extract_icd_code_and_with_llm(text):
    icd_model = OllamaLLM(model="mistral") # Change to the appropriate model for ICD code extraction
    prompt = (f'''
    Given the following medical information, extract up to 2 most relevant ICD-10 codes with their short names.
    The ICD-10 code generated should be most relevant to the medical information provided.
    The ICD-10 code and the short name should be in the universal format:
    If there is no relevant information, return an empty list.
    Return as JSON list: [{{"code": "ICD_CODE", "name": "Diagnosis Name"}}, ...]

    Example output:
    [
        {{"code": "E11.9", "name": "Type 2 diabetes mellitus without complications"}},
        {{"code": "I10", "name": "Essential (primary) hypertension"}}
    ]
    
    Do not include any other text or explanations, just the JSON list.
    

    [DATA]
    {text}
    ''')
    try:
        print(f"Prompt: {prompt}")
        res = icd_model.invoke(prompt)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()
        print(f"LLM Response: {cleaned_res}")

        cleaned_res = map_icd_codes_with_names(cleaned_res) # map the icd codes with the names in icd_codes_with_desc.json
        cleaned_res = json.dumps(cleaned_res, ensure_ascii=False)

        return cleaned_res
    except Exception as e:
        return f"LLM Error: {e}"


def summarizer():
    st.title("Amrita HIS Patient Cohort Summarizer")

    st.sidebar.subheader("Model Selection")
    available_models = [
        "mistral",
        "deepseek-r1",
        "deepseek-llm",
        "llama3.2",
        "qwen3"
    ]
    selected_model_name = st.sidebar.radio("Choose an LLM:", available_models)

    model = OllamaLLM(model=selected_model_name)

    # File uploader csv and excel
    uploaded_file = st.file_uploader("Upload a medical file", type=["csv", "xlsx"])

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully.")

        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
            st.success("Excel file uploaded successfully.")

        # Prompt input
        st.subheader("Prompt Template")
        prompt_options = {

            "Default":'''
            You are a clinical summarization assistant. Given structured or semi-structured patient data, generate a clear, detailed summary in a single paragraph suitable for medical professionals.
             Follow these guidelines:
             - Exclude any fields that are marked as 'Unknown', left empty, or not clinically relevant.
             - Do not assume or infer missing information; omit fields entirely if key clinical details (such as age, diagnosis, or medications) are unavailable.
             - Extract and present all clinically meaningful content, including diagnoses, treatment history, procedures, progress notes, lab results, and medications.
             - For medications, include dosage, route, frequency, and duration whenever available.
             - Present information in medically appropriate chronological order, emphasizing clinical progression and changes over time.
             - Maintain a professional, fluent tone using precise clinical language consistent with medical documentation standards.
             - Do not use bullet points, special characters, markdown symbols, or conversational language.
             - Ensure the output reads as a continuous, well-structured clinical note or discharge summary.
 
             Only return the paragraph itself — do not include headers, labels, commentary, closing statements or formatting instructions.
            ''',

            "Oncology": '''
                You are a clinical documentation specialist with expertise in oncology and internal medicine.
                Given the raw patient case data below, perform the following tasks:
                - Generate a clear, detailed summary in a single paragraph suitable for medical professionals.

                Generate a SOAP note with clear headings:
                - Subjective: Patient-reported symptoms, complaints, relevant history
                - Objective: Observations, vital signs, imaging, lab data
                - Assessment: Clinical interpretation, staging (if present), diagnosis
                - Plan: Treatment plans, medication changes, follow-up steps

                Convert the medication data into a clean bullet-point list:
                - Include drug name, dosage, route, and frequency if available

                Guidelines:
                Ensure clear headings for each section in the final output.
                Use bullet points where helpful, especially for Objective findings and the Plan.
                Normalize terminology and abbreviations to standard medical language.
                Do not use markdown or code formatting — just plain clinical text.

                Keep the output succinct, structured, and medically accurate, suitable for semantic search.
            ''',

            "Radiology": '''
                You are a medical assistant AI that reformats raw radiology report data into a clean and concise narrative suitable for semantic search and retrieval in a RAG system.
                From the input text, identify and rephrase any information corresponding to the following conceptual sections:
                Examination (what imaging was performed)
                Indication (why the study was done, such as symptoms or clinical concern)
                Findings (what was observed in the current imaging)
                Impression (overall summary of findings, if available)

                Your task is to:
                Combine the information into a natural, single-paragraph clinical note.
                Include section headers like "Examination:", "Findings:", etc.
                Omit any references to prior imaging or historical comparisons. Rewrite such content to describe only the current imaging status.
                Ignore demographic info, timestamps, referring physicians, or irrelevant metadata.
                Write in professional medical language suitable for downstream embedding in a vector database.
                                
            ''',

            "Custom": '''
            '''
        }

        selected_prompt_type = st.selectbox("Choose Prompt Type", options=list(prompt_options.keys()))

        # Use session state to allow prompt editing while preserving user input
        if "custom_prompt_text" not in st.session_state or st.session_state.get("last_prompt_type") != selected_prompt_type:
            st.session_state.custom_prompt_text = prompt_options[selected_prompt_type].strip()
            st.session_state.last_prompt_type = selected_prompt_type

        custom_prompt = st.text_area("Enter Custom Prompt (editable)", value=st.session_state.custom_prompt_text, height=400)
        prompt = custom_prompt.strip() if custom_prompt.strip() else prompt_options[selected_prompt_type].strip()


        if st.button("Summarize Records"):
            with st.spinner("Formatting rows..."):
                df["formatted_text"] = df.apply(format_row, axis=1)

            with st.spinner(f"Generating summaries with {selected_model_name}..."):
                df["summary"] = df["formatted_text"].apply(lambda x: summarize_with_llm(x, prompt, model))
            
            with st.spinner("Extracting ICD codes..."):
                df["icd_codes"] = df["summary"].apply(lambda x: extract_icd_code_and_with_llm(x))
            
            # create a new column with the summary and icd codes
            # df["summary_with_icd"] = df.apply(lambda x: f"{x['summary']}\nICD Codes: {x['icd_codes']}", axis=1)

            st.session_state["processed_df"] = df
            st.session_state["processed_filename"] = uploaded_file.name

        # download after processing
        if "processed_df" in st.session_state and st.session_state["processed_filename"] == uploaded_file.name:
            output_csv = st.session_state["processed_df"].to_csv(index=False).encode("utf-8")

            # extract the base name without extension
            file_name, _ = os.path.splitext(uploaded_file.name)
            new_file_name = f"{file_name}_summarized_{selected_model_name}.csv"

            st.download_button(
                label="Download Summarized CSV",
                data=output_csv,
                file_name=new_file_name,
                mime="text/csv"
            )