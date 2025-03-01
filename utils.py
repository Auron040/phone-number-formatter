import base64
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict

def get_download_link(df, filename, link_text):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def log_message(message):
    """Add a message to the log in the Streamlit app"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_message)
    print(log_message)  # Also print to console

def display_log():
    """Display the log messages in the Streamlit app"""
    if 'log_messages' in st.session_state and st.session_state.log_messages:
        for message in st.session_state.log_messages:
            st.text(message)
    else:
        st.text("No log messages yet.")

def init_session_state():
    """Initialize session state variables if they don't exist"""
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    
    if 'processed_dfs' not in st.session_state:
        st.session_state.processed_dfs = {}
    
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {}
    
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    
    if 'merge_stats' not in st.session_state:
        st.session_state.merge_stats = None
    
    if 'per_conflict_resolutions' not in st.session_state:
        st.session_state.per_conflict_resolutions = {}
    
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = []
