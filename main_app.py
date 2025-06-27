#main_app.py
import streamlit as st
from summarizer import summarizer 
from chatbot import chatbot





st.set_page_config(page_title="Amrita HIS Patient Cohort Dashboard", layout="wide")
st.sidebar.title("Cohort Insights Menu")
page = st.sidebar.selectbox("Please select whether you want to explore or summarize patient cohorts:", ["Amrita HIS Patient Cohort Explorer","Amrita HIS Patient Cohort Summarizer"])

if page == "Amrita HIS Patient Cohort Summarizer":
    summarizer()

elif page == "Amrita HIS Patient Cohort Explorer":
    # Inject CSS for custom dialog styling and button text-like appearance
    st.markdown(
        """
        <style>
        /* Customize dialog size */
        div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
            width: 80vw;
            height: 80vh;
            overflow: auto;
        }

        /* Base button style */
        button[kind="primary"] {
            width: 100%;
            background-color: #ffffff;       /* clean white */
            border: 2px solid #1e293b;       /* dark navy border */
            border-radius: 0;            /* pill shape */
            color: #1e293b;                   /* dark navy text */
            font-family: "Basier circle", -apple-system, system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            font-weight: 600 !important;
            font-size: 0.6rem !important;
            padding: 2px !important;
            box-shadow: 0 1px 4px rgba(30, 41, 59, 0.25);
            cursor: pointer;
            transition: background-color 0.25s ease, color 0.25s ease, box-shadow 0.25s ease;
            user-select: none;
            text-align: center;
            outline: none;
        }

        /* Hover effect */
        button[kind="primary"]:hover {
            background-color: #1e293b;       /* dark navy background */
            color: #ffffff;                  /* white text */
            box-shadow: 0 4px 12px rgba(30, 41, 59, 0.4);
        }

        /* Focus effect */
        button[kind="primary"]:focus {
            outline-offset: 2px;
            box-shadow: none;
        }

        /* Active / pressed effect */
        button[kind="primary"]:active {
            background-color: #15213a;
            box-shadow: 0 2px 6px rgba(21, 33, 58, 0.6);
        }



        
        /* Download button style */
        button[kind="secondary"] {
            padding: 0px 10px !important;
            font-size: 1rem !important;
            border: solid 1px #15213a !important;
            border-radius: 0 !important;
        }
        /* Hover */
        button[kind="secondary"]:hover {
            background-color: #15213a !important;
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(30, 41, 59, 0.4);
        }

        /* Focus */
        button[kind="secondary"]:focus {
            outline-offset: 2px;
            box-shadow: none;
            color: black !important;
        }


        /* Responsive font size for larger screens */
        @media (min-width: 768px) {
            button[kind="primary"] {
                font-size: 1.25rem;
                padding: 1rem 2.5rem;
            }
        }



        </style>
        """,
            unsafe_allow_html=True,
        )

    chatbot()
    