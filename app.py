import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
os.environ['PATH']+=os.pathsep+r"C:\Users\91763\Downloads\ffmpeg\ffmpeg\bin"
import cv2
from PIL import Image
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from wordcloud import WordCloud
import speech_recognition as sr
import time
import base64
import re
from collections import defaultdict, Counter
import seaborn as sns
from datetime import datetime, timedelta

# Import the enhanced summary components
from summary_assistant import RealTimeSummarizer, SummaryExplorer, PersonaSummarizer

st.set_page_config(
    page_title="Advanced Text, Video & Chat Summarizer",
    page_icon="📊",
    layout="wide"
)

# Try importing required packages with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    st.error("spaCy is not installed. Please install it using: pip install spacy")
    st.info("Some features like entity extraction will be limited.")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.error("pydub is not installed. Please install it using: pip install pydub")
    st.info("Audio extraction features will be limited.")

try:
    import pytube
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False
    st.error("pytube is not installed. Please install it using: pip install pytube")
    st.info("YouTube download features will be disabled.")

# Add custom CSS with updated styles for new features
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .chat-message {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .summary-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #43A047;
    }
    .highlight-text {
        background-color: #FFF9C4;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
    .realtime-update {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin-bottom: 1rem;
    }
    .persona-teacher {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
    }
    .persona-executive {
        background-color: #E8EAF6;
        border-left: 4px solid #3F51B5;
    }
    .persona-friend {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .persona-journalist {
        background-color: #E0F7FA;
        border-left: 4px solid #00BCD4;
    }
    .persona-sarcastic {
        background-color: #FCE4EC;
        border-left: 4px solid #EC407A;
    }
    .persona-steve_jobs {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .chat-input {
        border: 1px solid #BDBDBD;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .unseen-message {
        background-color: #FFECB3;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Import existing functions from your original app.py
# ... [Include your original functions here]
# For brevity, assuming these functions are in your existing app.py:
# check_spacy_model, load_summarization_model, load_spacy_model, summarize_text,
# extract_entities, simple_tokenize, generate_wordcloud, etc.

def check_spacy_model():
    if not SPACY_AVAILABLE:
        return False
        
    try:
        # Check if model is downloaded
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        st.error("spaCy language model not found. Please install it using: python -m spacy download en_core_web_sm")
        return False

# Load NLP models with error handling
@st.cache_resource
def load_summarization_model():
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSe