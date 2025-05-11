import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import random
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

# Set page config first before any other Streamlit command
st.set_page_config(
    page_title="Advanced Text, Video & Chat Summarizer",
    page_icon="📊",
    layout="wide"
)

# Import the enhanced summary components
from summary_assistant import RealTimeSummarizer, SummaryExplorer, PersonaSummarizer

# Initialize session state variables
if 'realtime_summarizer' not in st.session_state:
    st.session_state.realtime_summarizer = RealTimeSummarizer()
if 'summary_explorer' not in st.session_state:
    st.session_state.summary_explorer = SummaryExplorer()
if 'persona_summarizer' not in st.session_state:
    st.session_state.persona_summarizer = PersonaSummarizer()
if 'last_check_timestamp' not in st.session_state:
    st.session_state.last_check_timestamp = datetime.now()

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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        return summarizer, True
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None, False


def load_spacy_model():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy model: {e}")
        return None

# Text processing functions
def summarize_text(text, max_length=150, min_length=40, summarizer=None):
    if not summarizer:
        st.error("Summarization model not available.")
        return "Summary not available."
    
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Error generating summary."

def extract_entities(text, nlp=None):
    if not nlp:
        return []
    
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return []

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def generate_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=200, contour_width=3, contour_color='steelblue')
        wordcloud.generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

def analyze_sentiment(text, nlp=None):
    """Analyze sentiment using spaCy's TextBlob integration"""
    if not nlp:
        return "neutral", 0
    
    try:
        from spacytextblob.spacytextblob import SpacyTextBlob
        nlp.add_pipe("spacytextblob")
        doc = nlp(text)
        polarity = doc._.blob.polarity
        
        if polarity > 0.2:
            return "positive", polarity
        elif polarity < -0.2:
            return "negative", polarity
        else:
            return "neutral", polarity
    except Exception as e:
        return "neutral", 0

def extract_keywords(text, nlp=None, top_n=10):
    """Extract keywords using spaCy"""
    if not nlp:
        return []
    
    try:
        doc = nlp(text)
        # Filter for nouns and proper nouns that aren't stopwords
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
        # Count frequency
        keyword_freq = Counter(keywords)
        # Return top N
        return keyword_freq.most_common(top_n)
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

# Video processing functions
def process_video(video_file, frame_interval=30):
    """Extract frames and audio from video"""
    try:
        temp_video_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_video_path = temp_file.name
        
        # Open video file
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        
        # Extract audio if pydub is available
        audio_text = ""
        if PYDUB_AVAILABLE:
            try:
                temp_audio_path = temp_video_path.replace('.mp4', '.wav')
                
                # Extract audio
                command = f"ffmpeg -i {temp_video_path} -q:a 0 -map a {temp_audio_path} -y"
                os.system(command)
                
                # Process audio to text
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_audio_path) as source:
                    audio_data = recognizer.record(source)
                    audio_text = recognizer.recognize_google(audio_data)
                
                # Clean up
                os.remove(temp_audio_path)
            except Exception as e:
                st.warning(f"Audio extraction failed: {e}")
        
        # Clean up video file
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            
        return frames, audio_text, total_frames, fps
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return [], "", 0, 0

def download_youtube_video(youtube_url):
    """Download a YouTube video and return it as a file-like object"""
    if not PYTUBE_AVAILABLE:
        st.error("YouTube download feature is not available. Please install pytube.")
        return None
    
    try:
        yt = pytube.YouTube(youtube_url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        # Download to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
        
        # Return as file-like object
        with open(temp_file.name, 'rb') as f:
            video_data = f.read()
        
        # Clean up
        os.remove(temp_file.name)
        
        return video_data, yt.title
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None, ""

def generate_realtime_summary(context, new_text):
    """Generate a real-time summary as chat progresses"""
    if not context:
        return new_text
    
    try:
        # Logic to incrementally update summary
        # For simplicity, we'll use a weighted combination
        # In a real app, you might use a more sophisticated approach
        summarizer, _ = load_summarization_model()
        combined_text = f"{context}\n\nNEW INPUT: {new_text}"
        
        if len(combined_text.split()) > 100:
            return summarize_text(combined_text, max_length=150, min_length=50, summarizer=summarizer)
        else:
            return combined_text
    except Exception as e:
        st.error(f"Error generating real-time summary: {e}")
        return context + "\n\n" + new_text

def get_persona_summary(text, persona="neutral"):
    """Generate a summary in the style of a particular persona"""
    personas = {
        "teacher": "Summarize this in a clear, educational way with key lessons:",
        "executive": "Provide a concise executive summary with key action items:",
        "friend": "Explain this to me like I'm your friend, casually:",
        "journalist": "Summarize this as if writing a news article:",
        "sarcastic": "Summarize this in a sarcastic, humorous way:",
        "steve_jobs": "Summarize this as if you were Steve Jobs presenting a revolutionary product:"
    }
    
    try:
        summarizer, _ = load_summarization_model()
        prompt = personas.get(persona.lower(), "Summarize this:") + " " + text
        summary = summarize_text(prompt, max_length=150, min_length=50, summarizer=summarizer)
        return summary
    except Exception as e:
        st.error(f"Error generating persona summary: {e}")
        return f"Unable to generate {persona} summary: {str(e)}"

def update_chat_history(chat_history, message, is_user=True):
    """Update chat history with new message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    chat_history.append({
        "message": message,
        "is_user": is_user,
        "timestamp": timestamp,
        "seen": False if is_user else True
    })
    return chat_history

def get_chat_summary(chat_history):
    """Generate a summary of the entire chat history"""
    if not chat_history:
        return "No chat history to summarize."
    
    try:
        # Combine all messages
        full_text = ""
        for entry in chat_history:
            sender = "User" if entry["is_user"] else "Assistant"
            full_text += f"{sender}: {entry['message']}\n\n"
        
        # Generate summary
        summarizer, _ = load_summarization_model()
        summary = summarize_text(full_text, max_length=200, min_length=50, summarizer=summarizer)
        return summary
    except Exception as e:
        st.error(f"Error summarizing chat: {e}")
        return "Error generating chat summary."

def get_file_download_link(text, filename, link_text):
    """Generate a link to download text as a file"""
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

def main():
    st.markdown('<h1 class="main-header">Advanced Text, Video & Chat Summarizer</h1>', unsafe_allow_html=True)
    
    # Initialize models
    summarizer, summarizer_loaded = load_summarization_model()
    spacy_model = load_spacy_model() if check_spacy_model() else None
    
    # Initialize session state for chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = ""
    if 'realtime_summary' not in st.session_state:
        st.session_state.realtime_summary = ""
    if 'user_input' not in st.session_state:  # For storing user input value
        st.session_state.user_input = ""
    
    # Main tabs
    tabs = st.tabs(["Text Summarizer", "Video Summarizer", "Chat Summarizer", "About"])
    
    # [... keep all the code for Text Summarizer and Video Summarizer tabs unchanged ...]
    
    # Chat Summarizer Tab
    with tabs[2]:
        st.markdown('<h2 class="sub-header">Chat Summarizer</h2>', unsafe_allow_html=True)
        
        # Chat interface
        chat_container = st.container()
        
        # Input area - Use a key different from the session state variable
        user_input = st.text_input("Type your message:", key="chat_input_widget", value=st.session_state.user_input)
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            if st.button("Send Message"):
                if user_input:
                    # Add user message to chat history
                    st.session_state.chat_history = update_chat_history(
                        st.session_state.chat_history, user_input, is_user=True
                    )
                    
                    # Generate a simple response
                    response = f"I received your message: {user_input}"
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history = update_chat_history(
                        st.session_state.chat_history, response, is_user=False
                    )
                    
                    # Update real-time summary
                    st.session_state.realtime_summary = generate_realtime_summary(
                        st.session_state.realtime_summary, user_input
                    )
                    
                    # Clear input by updating session state variable
                    st.session_state.user_input = ""
        
        # [... keep the rest of the Chat Summarizer tab code unchanged ...]
    
    # [... keep the About tab code unchanged ...]
     with tabs[3]:
        st.markdown('<h2 class="sub-header">About This App</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Advanced Text, Video & Chat Summarizer
        
        This application provides tools for summarizing and analyzing:
        
        - **Text documents** - Paste text, upload files, or use our examples
        - **Video content** - Upload videos or provide YouTube URLs
        - **Chat conversations** - Have a conversation and get real-time summaries
        
        ### Features
        
        - Multiple summarization styles (concise, detailed, bullets, persona-based)
        - Entity extraction and visualization
        - Sentiment analysis
        - Keyword identification
        - Video frame extraction and analysis
        - Audio transcription from videos
        - Real-time chat summarization
        - Multi-persona perspectives
        
        ### How to Use
        
        1. Select a tab for the type of content you want to analyze
        2. Follow the prompts to input your content
        3. Choose analysis options
        4. View and download results
        
        ### Technologies
        
        This app uses several state-of-the-art NLP and computer vision technologies:
        
        - **Transformers** (BART) for text summarization
        - **spaCy** for entity extraction and NLP tasks
        - **OpenCV** for video processing
        - **Speech Recognition** for audio transcription
        - **WordCloud** for text visualization
        """)
        
        # Stats about current session
        st.subheader("Session Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chats Processed", len(st.session_state.chat_history))
        with col2:
            # Check how long session has been active
            session_start = st.session_state.get('session_start', datetime.now())
            if 'session_start' not in st.session_state:
                st.session_state.session_start = session_start
            
            session_duration = datetime.now() - st.session_state.session_start
            st.metric("Session Duration", f"{session_duration.seconds // 60} minutes")
        with col3:
            st.metric("Summaries Generated", 
                     len([1 for entry in st.session_state.chat_history if not entry.get("is_user", True)]))
        
        # Show system status
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Summarization Model:** {'✅ Loaded' if summarizer_loaded else '❌ Not Loaded'}")
            st.markdown(f"**spaCy NLP Model:** {'✅ Loaded' if spacy_model else '❌ Not Loaded'}")
        
        with col2:
            st.markdown(f"**YouTube Support:** {'✅ Available' if PYTUBE_AVAILABLE else '❌ Not Available'}")
            st.markdown(f"**Audio Processing:** {'✅ Available' if PYDUB_AVAILABLE else '❌ Not Available'}")

# Check for realtime updates in a chat
def check_for_unseen_messages():
    """Function to check for unseen messages and update notifications"""
    # For demo purposes only - in a real app this would connect to a backend
    current_time = datetime.now()
    time_diff = current_time - st.session_state.last_check_timestamp
    
    # Check every 10 seconds in this demo
    if time_diff.seconds >= 10:
        st.session_state.last_check_timestamp = current_time
        
        # In a real app, this would check a database or API
        # For demo, we'll just return True randomly
        if random.random() > 0.7:  # 30% chance of new message
            new_msg = "This is a simulated notification message. In a real app, this would be from your database."
            st.session_state.chat_history = update_chat_history(
                st.session_state.chat_history, new_msg, is_user=False
            )
            return True
    
    return False

# Run the app
if __name__ == "__main__":
    main()
    # In a real app, you might want to use Streamlit's experimental_rerun or callbacks
    # to periodically check for new messages
    check_for_unseen_messages()