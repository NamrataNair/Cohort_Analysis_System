# chatbot.py
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import pandas as pd
import os
import re
import json
import pickle

from bs4 import BeautifulSoup
from streamlit import dialog
import time



@dialog("Patient Record")
def show_patient_record(record_id, content):

    st.html(f"<div class='big-dialog'><h4>Summary for ID {record_id}</h4><p>{content}</p></div>")




# Summary part 

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


def summarize_with_llm_chatbot(text):

    prompt = '''
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
        '''
    
    prompt = (f'''
    {prompt}
    
    [DATA]
    {text}
    ''')
    model_summary_chatbot = OllamaLLM(model="mistral") # Change to the appropriate model for summarization

    try:
        print(f"Chatbot Summary Prompt: {prompt}")
        res = model_summary_chatbot.invoke(prompt)
        cleaned_res = re.sub(r'<think>.*?</think>', '', res, flags=re.DOTALL).strip()

        cleaned_res = clean_markdown_from_llm(cleaned_res)

        print(f"LLM Response: {cleaned_res}")
        return cleaned_res
    except Exception as e:
        return f"LLM Error: {e}"

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
    

def extract_icd_code_and_with_llm_chatbot(text):
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






# chatbot 

# Extracts ICD codes with their descriptions from a JSON string to a list of strings
# For the ICD codes in the document metadata
# Example input: '[{"code": "C18.9", "name": "Malignant neoplasm of colon, unspecified"}, {"code": "C81.02", "name": "Hodgkin lymphoma, lymphocyte depleted, unspecified site"}]'
# Example output: ['C18.9 : Malignant neoplasm of colon, unspecified', 'C81.02 : Hodgkin lymphoma, lymphocyte depleted, unspecified site']
def extract_icd_codes(icd_code_str):
    try:
        icd_list = json.loads(icd_code_str)

        res =  [
            f'{entry["code"]} : {entry["name"]}'
            for entry in icd_list
            if "code" in entry and "name" in entry
        ]
        # print(res)
        return res
    except Exception:
        return []

# Parse and format ICD codes into a markdown table
# From the retrieved documents metadata -> icd_codes_with_name
def format_icd_codes_table(icd_codes_retrieved):
    if not icd_codes_retrieved:
        return "No ICD codes found in the retrieved documents."

    table_header = "| ICD Code | Description |\n|----------|-------------|"
    table_rows = []

    for icd_code_str in icd_codes_retrieved:
        try:
            icd_list = json.loads(icd_code_str) if isinstance(icd_code_str, str) else icd_code_str
            for icd in icd_list:
                code = icd.get("code", "").strip()
                name = icd.get("name", "").strip()
                if code and name:
                    table_rows.append(f"| {code} | {name} |")
        except json.JSONDecodeError:
            continue

    if not table_rows:
        return "No valid ICD codes found in the retrieved documents."

    return table_header + "\n" + "\n".join(table_rows)


def chatbot():
    st.title("Amrita HIS Patient Cohort Explorer")

    if "db_updated" not in st.session_state:
        st.session_state.db_updated = False

    with st.sidebar:
        st.header("Patient Cohort Data Manager")
        base_dir = "./chroma_langchain_db"
        os.makedirs(base_dir, exist_ok=True)

        existing_dbs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
        if st.session_state.db_updated:
            existing_dbs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
            st.session_state.db_updated = False

        # db_action = st.selectbox("Choose an option:", ["Select Existing File", "Upload File"])
        db_action = st.selectbox(
        "Start by selecting or uploading a patient cohort:",
        ["Select an Existing Patient Cohort", "Upload a New Patient Cohort"]
        )

        vector_store = None
        retriever = None

        if db_action == "Upload a New Patient Cohort":
            new_db_name = st.text_input("Enter a name for the cohort:", placeholder="e.g., Oncology_Records")
            if new_db_name:
                db_path = os.path.join(base_dir, new_db_name)
                if os.path.exists(db_path):
                    st.error(f"A cohort named '{new_db_name}' available.")
                    new_db_name = ""

            uploaded_file = st.file_uploader("To create a new patient cohort, please upload a file.", type=["csv", "xlsx"])

            if new_db_name and uploaded_file:
                if st.button("Upload"):

                    # df = pd.read_csv(uploaded_file)

                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                        st.success("CSV file uploaded successfully.")

                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                        st.success("Excel file uploaded successfully.")
                    else:
                        st.error("Unsupported file format. Please upload a CSV or Excel file.")
                        return

                    if "summary" not in df.columns:
                        with st.spinner("Formatting rows..."):
                            df["formatted_text"] = df.apply(format_row, axis=1)
                        
                        with st.spinner(f"Generating summaries..."):
                            df["summary"] = df["formatted_text"].apply(lambda x: summarize_with_llm_chatbot(x))

                        with st.spinner("Extracting ICD codes..."):
                            df["icd_codes"] = df["summary"].apply(lambda x: extract_icd_code_and_with_llm_chatbot(x))
            
                    # print first 5 rows of the dataframe
                    print(df.head())

                    st.success(f"Creating patient cohort file : {new_db_name}")
                    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

                    documents = []
                    ids = []
                    for i, row in df.iterrows():
                        
                        icd_codes = extract_icd_codes(row.get("icd_codes", "[]"))
                        #eg: icd_codes = ["C18.9 : Malignant neoplasm of colon, unspecified", "C81.02 : Hodgkin lymphoma, lymphocyte depleted, unspecified site"]
                        # print("---------")
                        documents.append(Document(
                            page_content=row["summary"],
                            metadata={

                                "record_id": str(i+2), # i : start fom 0, excel row starts from 1, row 1 is header, so +2
                                "icd_code_1": icd_codes[0] if len(icd_codes) > 0 else "",
                                "icd_code_2": icd_codes[1] if len(icd_codes) > 1 else "",
                                "icd_codes_with_name" : row["icd_codes"] if "icd_codes" in row else ""
                                }
                                ))
                                
                        ids.append(str(i))

                    vector_store = Chroma(
                        collection_name="patient_records",
                        persist_directory=db_path,
                        embedding_function=embeddings
                    )
                    vector_store.add_documents(documents=documents, ids=ids)

                    '''
                    changes possible (optional)
                    1. retriever = vectorstore.as_retriever(search_type="mmr")
                    2. k = a different number
                    '''

                    # Here no need of retriever, since we are not using it, 
                    # This dropdown is uplading a new file, so no need of retriever
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 

                    st.session_state.db_updated = True
                    st.rerun()

        elif db_action == "Select an Existing Patient Cohort":

            # set fiest option as empty string
            existing_dbs = [""] + existing_dbs
            
            selected_db = st.selectbox("Select a patient cohort you have already uploaded:", existing_dbs)
            if selected_db:
                st.success(f"Patient cohort ‘{selected_db}’ is now ready to use.")
                db_path = os.path.join(base_dir, selected_db)
                embeddings = OllamaEmbeddings(model="mxbai-embed-large")
                vector_store = Chroma(
                    collection_name="patient_records",
                    persist_directory=db_path,
                    embedding_function=embeddings
                )

                # Extract all ICD codes for showing in the selectbox

                all_icd_codes = []
                try:
                    # Load the key-value list from the pickle file
                    # ['69.8 : Other specified spirochaetal infections', ...]
                    with open('key_value_list.pkl', 'rb') as f:
                        loaded_key_value_list = pickle.load(f)
                        all_icd_codes = loaded_key_value_list
                    

                except Exception as e:
                    st.warning("Could not extract ICD codes from metadata.")

                selected_icd_code = st.selectbox("Filter patient records by ICD Code:", ["All"] + sorted(all_icd_codes))

                # use sorted ICD codes  (above line) , below line for testing purpose 
                # selected_icd_code = st.selectbox("Select ICD Code:", ["All"] + list(all_icd_codes))

                filter_dict = {}
                if selected_icd_code != "All":
                    filter_dict = {
                        "$or": [
                            {"icd_code_1": selected_icd_code},
                            {"icd_code_2": selected_icd_code},
                        ]
                    }

                    '''
                    changes possible (optional)
                    1. retriever = vectorstore.as_retriever(search_type="mmr")
                    2. k = a different number
                    3. filter = filter_dict can improve the search
                    '''
                    retriever = vector_store.as_retriever(
                        search_kwargs={
                            "k": 5,
                            "filter": filter_dict
                        }
                    )
                else:
                    '''
                    changes possible (optional)
                    1. retriever = vectorstore.as_retriever(search_type="mmr")
                    2. k = a different number
                    '''
                    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

                
            else:
                st.info("Please select a patient cohort to proceed.")

            st.subheader("Selected Patient Cohort")
            st.write(selected_db if selected_db else "No patient cohort selected.")

        else:
            st.info("Select an action to get started.")
        


    if not retriever:
        st.warning("Please select or upload a patient cohort to start using the chatbot.")
        return


    # Model - Change here to use different models 
    model = OllamaLLM(model="mistral") # Model name can be changed to any other model name
    template = """
        You are a medical AI assistant specializing in clinical diagnosis, treatment planning and patient case analysis.
        Use only the provided patient records to answer the question.
        If any required information is missing or unclear, state it explicitly. Do not make assumptions or use external knowledge.
        Create a detailed, clear and concise response based on the patient records provided.

        If no relevant patient records are found, respond with "No relevant patient records found.".
        If the question is not related to the patient records, respond with "The question is not related to the patient records.".
    --- 

    Patient Records:
    {patient_data}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # record the chat history
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []


    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧬"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown("**Matching Rows**")

                if isinstance(msg["content"], dict):
                    # Show record_id buttons in rows of 5 columns per row
                    record_items = list(msg["content"]["patient_data_id_dict"].items())
                    buttons_per_row = 5
                    from math import ceil
                    num_rows = ceil(len(record_items) / buttons_per_row)

                    for row in range(num_rows):
                        cols = st.columns(buttons_per_row)
                        for i in range(buttons_per_row):
                            idx = row * buttons_per_row + i
                            if idx >= len(record_items):
                                break
                            record_id, content = record_items[idx]
                            with cols[i]:
                                if st.button(f"{record_id}", key=f"history_view_{record_id}_{hash(str(msg))}", type="primary"):
                                    show_patient_record(record_id, content)

                    st.markdown(msg["content"]["cleaned_res"])
                else:
                    st.markdown(msg["content"])
        st.empty()

        

    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # record the chat history
        # st.session_state.chat_log.append({"role": "user", "content": user_input})

        with st.chat_message("user", avatar="🧬"):
            st.markdown(user_input)

        with st.spinner("Analyzing data..."):


            start_time = time.time()  # Start the timer

            retrieved_docs = retriever.invoke(user_input)  # Your retrieval call

            end_time = time.time()  # End the timer
            retrieval_duration = end_time - start_time  # Time in seconds


            patient_data = [doc.page_content for doc in retrieved_docs]

            print("patient_data")
            print(patient_data)


            document_ids = [doc.metadata["record_id"] for doc in retrieved_docs]

            # create a dictionary with key as document ID and value as the corresponding patient data
            patient_data_id_dict = {doc.metadata["record_id"]: doc.page_content for doc in retrieved_docs}

            patient_data = "\n".join(patient_data)
            document_ids = ", ".join(document_ids)


            # icd_codes_retrieved = []
            # for doc in retrieved_docs:
            #     icd_codes_retrieved.append(doc.metadata["icd_codes_with_name"])
            # icd_codes_retrieved = "\n\n".join(icd_codes_retrieved)

            if len(patient_data) > 0:
                icd_codes_retrieved = []
                for doc in retrieved_docs:
                    icd_codes_retrieved.append(doc.metadata["icd_codes_with_name"])  

                icd_codes_table = format_icd_codes_table(icd_codes_retrieved)

                result = chain.invoke({"patient_data": patient_data, "question": user_input})
                cleaned_res = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

                # cleaned_res = f"""**Matching Rows** : {document_ids}\n\n**ICD Codes**\n\n{icd_codes_table}\n\n**Response**\n\n{cleaned_res}"""
                cleaned_res = f"""**ICD Codes**\n\n{icd_codes_table}\n\n**Response**\n\n{cleaned_res}"""
            else:
                cleaned_res = "No relevant patient records found."
            
            cleaned_res_with_summary_id = {}

            cleaned_res_with_summary_id["patient_data_id_dict"]  = patient_data_id_dict
            cleaned_res_with_summary_id["cleaned_res"] = cleaned_res

        st.session_state.messages.append({"role": "assistant", "content": cleaned_res_with_summary_id})

        # record the chat history
        st.session_state.chat_log.append({"user": user_input, "assistant": cleaned_res_with_summary_id["cleaned_res"],"document_ids": document_ids, "retrival_duration": retrieval_duration})


        
        with st.chat_message("assistant"):

            st.markdown("**Matching Rows**")

            # Expand for id and show the content
            record_items = list(cleaned_res_with_summary_id["patient_data_id_dict"].items())
            cols = st.columns(5)  # 5 buttons per row, you can change this

            for idx, (record_id, content) in enumerate(record_items):
                col = cols[idx % 5]  # Place each button in one of the 5 columns
                with col:
                    if st.button(f"{record_id}", key=f"new_view_{record_id}", type="primary"):
                        show_patient_record(record_id, content)



            st.markdown(cleaned_res_with_summary_id["cleaned_res"])


            st.empty()

    if st.session_state.chat_log:

        # Input for the user to enter a filename as a popup



        chat_text = "\n\n".join(
            [
                f"User: {entry['user']}\n Document IDs: {entry['document_ids']}\n Retrieval Duration: {entry['retrival_duration']:.2f} seconds\n Assistant: {entry['assistant']}"                    for entry in st.session_state.chat_log
            ]
        )

        # st.download_button(
        #     label="Export Chat Log",
        #     data=chat_text.encode("utf-8"),
        #     file_name="chat_log.txt",
        #     mime="text/plain"
        # )

        @dialog("Export Chat Log")
        def export_chat_log():
            filename = st.text_input("Enter a filename:", placeholder="chat_log")
            st.download_button(
                label="Download",
                data=chat_text.encode("utf-8"),
                file_name=f"{filename}.txt",
                mime="text/plain",
                key="download_btn"
            )

        
        if st.button("Download Chat Log", type="secondary"):
            export_chat_log()



