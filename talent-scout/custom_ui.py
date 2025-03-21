import streamlit as st
import os

def load_custom_css():
    # Load Bootstrap CSS from a CDN
    bootstrap_link = """
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    """
    st.markdown(bootstrap_link, unsafe_allow_html=True)
    
    # Load custom CSS from the static folder (if it exists)
    css_path = os.path.join("static", "custom.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            custom_css = f.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

def load_advanced_ui():
    # Additional advanced styling for the UI
    advanced_css = """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    .chat-message {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .chat-message.user {
        background-color: #d1e7dd;
        text-align: right;
    }
    .chat-message.assistant {
        background-color: #f8d7da;
        text-align: left;
    }
    </style>
    """
    st.markdown(advanced_css, unsafe_allow_html=True)

def st_phone_number(label, placeholder="", default_country="CN"):
    """
    A simple phone number input wrapper.
    You can extend this function for more advanced behavior.
    """
    return st.text_input(label, placeholder=placeholder)
