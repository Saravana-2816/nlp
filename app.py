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
            st.info("No entries to display.")
            st.error("No entries to display.")
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
    """Download a YouTube video and return its path"""
    if not PYTUBE_AVAILABLE:
        st.error("YouTube download functionality is not available. Please install pytube.")
        return None
    
    try:
        st.info("Downloading YouTube video... This may take a moment.")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download the video
        youtube = pytube.YouTube(youtube_url)
        video = youtube.streams.get_highest_resolution()
        video_path = video.download(temp_dir)
        
        # Return the video file
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        # Clean up
        os.remove(video_path)
        os.rmdir(temp_dir)
        
        return video_bytes
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# Chat data processing
def analyze_chat_data(chat_text):
    """Analyze uploaded chat data (from WhatsApp, Slack, etc.)"""
    try:
        # Try to determine format (WhatsApp, Slack, Teams, etc.)
        format_type = detect_chat_format(chat_text)
        
        if format_type == "whatsapp":
            messages = parse_whatsapp_chat(chat_text)
        elif format_type == "slack":
            messages = parse_slack_chat(chat_text)
        elif format_type == "teams":
            messages = parse_teams_chat(chat_text)
        else:
            messages = parse_generic_chat(chat_text)
        
        return messages, format_type
    except Exception as e:
        st.error(f"Error analyzing chat data: {e}")
        return [], "unknown"

def detect_chat_format(chat_text):
    """Detect the chat format based on patterns"""
    lines = chat_text.split('\n')[:20]  # Check first 20 lines
    
    # WhatsApp format: [MM/DD/YY, HH:MM:SS] Name: Message
    whatsapp_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\]\s[^:]+:'
    whatsapp_matches = sum(1 for line in lines if re.match(whatsapp_pattern, line))
    
    # Slack format: [Time] User: Message
    slack_pattern = r'\[\d{1,2}:\d{2}(?::\d{2})?\]\s[^:]+:'
    slack_matches = sum(1 for line in lines if re.match(slack_pattern, line))
    
    # Teams format: [Name] [Time]: Message
    teams_pattern = r'\[[^\]]+\]\s\[\d{1,2}:\d{2}(?::\d{2})?\]:'
    teams_matches = sum(1 for line in lines if re.match(teams_pattern, line))
    
    # Determine format based on matches
    if whatsapp_matches > max(slack_matches, teams_matches):
        return "whatsapp"
    elif slack_matches > teams_matches:
        return "slack"
    elif teams_matches > 0:
        return "teams"
    else:
        return "generic"

def parse_whatsapp_chat(chat_text):
    """Parse WhatsApp chat format"""
    messages = []
    pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)\]\s([^:]+):\s(.*)'
    
    current_msg = ""
    current_sender = ""
    current_time = None
    
    for line in chat_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            # If we have a message in progress, save it
            if current_sender:
                messages.append({
                    "timestamp": current_time,
                    "sender": current_sender,
                    "message": current_msg.strip()
                })
            
            # Start a new message
            date_str, time_str, sender, message = match.groups()
            try:
                date_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M:%S")
            except ValueError:
                try:
                    date_obj = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M")
                except ValueError:
                    date_obj = datetime.now()
            
            current_time = date_obj
            current_sender = sender.strip()
            current_msg = message
        else:
            # Continuation of previous message
            if current_sender:
                current_msg += "\n" + line
    
    # Add the last message
    if current_sender:
        messages.append({
            "timestamp": current_time,
            "sender": current_sender,
            "message": current_msg.strip()
        })
    
    return messages

def parse_slack_chat(chat_text):
    """Parse Slack chat format"""
    messages = []
    pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s([^:]+):\s(.*)'
    
    current_msg = ""
    current_sender = ""
    current_time = None
    
    for line in chat_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            # If we have a message in progress, save it
            if current_sender:
                messages.append({
                    "timestamp": current_time,
                    "sender": current_sender,
                    "message": current_msg.strip()
                })
            
            # Start a new message
            time_str, sender, message = match.groups()
            try:
                # Use today's date since Slack format usually doesn't include date
                date_obj = datetime.combine(datetime.today().date(), 
                                           datetime.strptime(time_str, "%H:%M:%S").time())
            except ValueError:
                try:
                    date_obj = datetime.combine(datetime.today().date(),
                                              datetime.strptime(time_str, "%H:%M").time())
                except ValueError:
                    date_obj = datetime.now()
            
            current_time = date_obj
            current_sender = sender.strip()
            current_msg = message
        else:
            # Continuation of previous message
            if current_sender:
                current_msg += "\n" + line
    
    # Add the last message
    if current_sender:
        messages.append({
            "timestamp": current_time,
            "sender": current_sender,
            "message": current_msg.strip()
        })
    
    return messages

def parse_teams_chat(chat_text):
    """Parse Microsoft Teams chat format"""
    messages = []
    pattern = r'\[([^\]]+)\]\s\[(\d{1,2}:\d{2}(?::\d{2})?)\]:\s(.*)'
    
    current_msg = ""
    current_sender = ""
    current_time = None
    
    for line in chat_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            # If we have a message in progress, save it
            if current_sender:
                messages.append({
                    "timestamp": current_time,
                    "sender": current_sender,
                    "message": current_msg.strip()
                })
            
            # Start a new message
            sender, time_str, message = match.groups()
            try:
                # Use today's date since Teams format usually doesn't include date
                date_obj = datetime.combine(datetime.today().date(), 
                                           datetime.strptime(time_str, "%H:%M:%S").time())
            except ValueError:
                try:
                    date_obj = datetime.combine(datetime.today().date(),
                                              datetime.strptime(time_str, "%H:%M").time())
                except ValueError:
                    date_obj = datetime.now()
            
            current_time = date_obj
            current_sender = sender.strip()
            current_msg = message
        else:
            # Continuation of previous message
            if current_sender:
                current_msg += "\n" + line
    
    # Add the last message
    if current_sender:
        messages.append({
            "timestamp": current_time,
            "sender": current_sender,
            "message": current_msg.strip()
        })
    
    return messages

def parse_generic_chat(chat_text):
    """Fallback generic chat parser"""
    messages = []
    # Simple line by line approach
    for i, line in enumerate(chat_text.split('\n')):
        if line.strip():
            messages.append({
                "timestamp": datetime.now() - timedelta(minutes=len(chat_text.split('\n'))-i),
                "sender": "Unknown",
                "message": line.strip()
            })
    
    return messages

def visualize_chat_metrics(messages):
    """Generate visualizations for chat analysis"""
    if not messages:
        return None, None, None
    
    # Prepare data
    senders = [msg["sender"] for msg in messages]
    timestamps = [msg["timestamp"] for msg in messages]
    message_lengths = [len(msg["message"]) for msg in messages]
    
    # Create dataframe
    df = pd.DataFrame({
        'sender': senders,
        'timestamp': timestamps,
        'message_length': message_lengths
    })
    
    # Messages per person
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sender_counts = df['sender'].value_counts()
    sns.barplot(x=sender_counts.index, y=sender_counts.values, ax=ax1)
    ax1.set_title('Messages per Person')
    ax1.set_ylabel('Number of Messages')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Messages over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df['date'] = df['timestamp'].dt.date
    messages_per_day = df.groupby('date').size()
    messages_per_day.plot(ax=ax2)
    ax2.set_title('Messages per Day')
    ax2.set_ylabel('Number of Messages')
    
    # Average message length per person
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    avg_length = df.groupby('sender')['message_length'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_length.index, y=avg_length.values, ax=ax3)
    ax3.set_title('Average Message Length per Person')
    ax3.set_ylabel('Average Characters')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    return fig1, fig2, fig3

# Main application layout
def main():
    # Initialize session state for real-time features
    if 'realtime_summarizer' not in st.session_state:
        st.session_state.realtime_summarizer = RealTimeSummarizer()
    
    if 'summary_explorer' not in st.session_state:
        st.session_state.summary_explorer = SummaryExplorer()
    
    if 'persona_summarizer' not in st.session_state:
        st.session_state.persona_summarizer = PersonaSummarizer()
    
    if 'last_check_timestamp' not in st.session_state:
        st.session_state.last_check_timestamp = datetime.now()
    
    # Title and description
    st.markdown('<h1 class="main-header">Advanced Text, Video & Chat Summarizer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This tool helps you summarize and analyze content from various sources:
    <ul>
        <li>📝 <b>Text</b>: Paste any text for instant summarization and analysis</li>
        <li>🎥 <b>Video</b>: Upload videos or use YouTube links to extract and summarize content</li>
        <li>💬 <b>Chat</b>: Upload chat exports to analyze conversations and generate insights</li>
    </ul>
    Plus new advanced features: real-time summaries, persona-based summaries, and interactive exploration!
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    summarization_model, model_loaded = load_summarization_model()
    nlp = load_spacy_model() if check_spacy_model() else None
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Text Summarizer", "Video Summarizer", "Chat Analyzer", "Real-Time Session"])
    
    # --------------------- TAB 1: TEXT SUMMARIZER ---------------------
    with tab1:
        st.markdown('<h2 class="sub-header">Text Summarization & Analysis</h2>', unsafe_allow_html=True)
        
        # Input area
        input_method = st.radio("Choose input method:", ["Enter Text", "Upload File"], horizontal=True)
        
        text_input = ""
        
        if input_method == "Enter Text":
            text_input = st.text_area("Enter text to summarize:", height=250)
        else:
            uploaded_file = st.file_uploader("Upload a text file:", type=['txt', 'md', 'pdf'])
            if uploaded_file:
                try:
                    # For PDF files
                    if uploaded_file.name.endswith('.pdf'):
                        try:
                            import PyPDF2
                            pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
                            text_list = []
                            for page_num in range(pdf_reader.numPages):
                                page = pdf_reader.getPage(page_num)
                                text_list.append(page.extractText())
                            text_input = "\n".join(text_list)
                        except Exception as e:
                            st.error(f"Error reading PDF: {e}")
                    else:
                        # For text files
                        text_input = uploaded_file.getvalue().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Process the text if there's input
        if text_input:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Summary")
                
                summary_options = st.expander("Summary Options", expanded=False)
                with summary_options:
                    max_length = st.slider("Maximum summary length:", 50, 500, 150)
                    min_length = st.slider("Minimum summary length:", 10, 100, 40)
                    
                    # Persona-based summaries
                    if st.checkbox("Use persona-based summary style"):
                        personas = st.session_state.persona_summarizer.get_available_personas()
                        persona_options = {persona: st.session_state.persona_summarizer.get_persona_description(persona) 
                                        for persona in personas}
                        
                        selected_persona = st.selectbox(
                            "Choose summary style:",
                            options=list(persona_options.keys()),
                            format_func=lambda x: f"{x.capitalize()} - {persona_options[x]}"
                        )
                        
                        # Generate persona-based summary
                        summary = st.session_state.persona_summarizer.summarize_with_persona(
                            text_input, selected_persona, max_length=max_length
                        )
                        
                        # Display with appropriate styling
                        st.markdown(f'<div class="persona-{selected_persona}">{summary}</div>', unsafe_allow_html=True)
                    else:
                        # Generate standard summary
                        if model_loaded:
                            summary = summarize_text(text_input, max_length=max_length, min_length=min_length, 
                                                   summarizer=summarization_model)
                            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                        else:
                            st.error("Summarization model could not be loaded. Please check your internet connection.")
                
                # Text statistics
                st.markdown("### Text Statistics")
                word_count = len(re.findall(r'\b\w+\b', text_input))
                sentence_count = len(re.findall(r'[.!?]+', text_input)) or 1  # Avoid division by zero
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                col_stats1.metric("Word Count", word_count)
                col_stats2.metric("Character Count", len(text_input))
                col_stats3.metric("Average Words per Sentence", round(word_count / sentence_count, 1))
                
                # Sentiment Analysis if spaCy is available
                if nlp:
                    try:
                        sentiment, polarity = analyze_sentiment(text_input, nlp)
                        st.markdown(f"**Sentiment:** {sentiment.capitalize()} ({polarity:.2f})")
                        
                        # Visual sentiment indicator
                        sentiment_color = "#4CAF50" if sentiment == "positive" else "#F44336" if sentiment == "negative" else "#9E9E9E"
                        st.markdown(f"""
                        <div style="background-color: {sentiment_color}; height: 10px; width: {abs(polarity) * 100}%; border-radius: 5px;"></div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error analyzing sentiment: {e}")
            
            with col2:
                # Word cloud
                st.markdown("### Word Cloud")
                wordcloud_fig = generate_wordcloud(text_input)
                if wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                
                # Key entities
                if nlp:
                    st.markdown("### Key Entities")
                    entities = extract_entities(text_input, nlp)
                    
                    if entities:
                        entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                        entity_types = entity_df["Type"].value_counts()
                        
                        # Entity count by type
                        fig, ax = plt.subplots(figsize=(10, 5))
                        entity_types.plot(kind="bar", ax=ax)
                        ax.set_title("Entity Types")
                        ax.set_ylabel("Count")
                        st.pyplot(fig)
                        
                        # Entity list
                        st.markdown("#### Named Entities Found:")
                        for entity_type in entity_df["Type"].unique():
                            entities_of_type = entity_df[entity_df["Type"] == entity_type]["Entity"].unique()
                            if len(entities_of_type) > 0:
                                st.markdown(f"**{entity_type}:** {', '.join(entities_of_type[:5])}" + 
                                          (f" and {len(entities_of_type) - 5} more..." if len(entities_of_type) > 5 else ""))
                    else:
                        st.info("No named entities found in the text.")
                
                # Keywords
                st.markdown("### Keywords")
                keywords = extract_keywords(text_input, nlp)
                if keywords:
                    # Create horizontal bar chart for top keywords
                    fig, ax = plt.subplots(figsize=(10, 5))
                    keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Count"])
                    sns.barplot(x="Count", y="Keyword", data=keyword_df, ax=ax)
                    ax.set_title("Top Keywords")
                    st.pyplot(fig)
                else:
                    st.info("No keywords extracted.")

    # --------------------- TAB 2: VIDEO SUMMARIZER ---------------------
    with tab2:
        st.markdown('<h2 class="sub-header">Video Summarization</h2>', unsafe_allow_html=True)
        
        video_source = st.radio("Select video source:", ["Upload Video", "YouTube URL"], horizontal=True)
        
        video_file = None
        
        if video_source == "Upload Video":
            video_file = st.file_uploader("Upload a video file:", type=['mp4', 'mov', 'avi', 'mkv'])
        else:
            youtube_url = st.text_input("Enter YouTube URL:")
            if youtube_url and PYTUBE_AVAILABLE:
                if st.button("Process YouTube Video"):
                    with st.spinner("Downloading YouTube video..."):
                        video_bytes = download_youtube_video(youtube_url)
                        if video_bytes:
                            # Save as a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                                temp_file.write(video_bytes)
                                temp_path = temp_file.name
                            
                            # Create a file-like object from the bytes
                            import io
                            video_file = io.BytesIO(video_bytes)
                            video_file.name = "youtube_video.mp4"
        
        if video_file:
            # Show the video
            st.video(video_file)
            
            # Process button
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    # Process the video
                    frames, audio_text, total_frames, fps = process_video(video_file)
                    
                    if frames:
                        st.success(f"Video processed! Extracted {len(frames)} frames.")
                        
                        # Display audio transcript if available
                        if audio_text:
                            st.markdown("### Audio Transcript")
                            st.markdown(f'<div class="info-text">{audio_text}</div>', unsafe_allow_html=True)
                            
                            # Summarize audio transcript
                            if model_loaded:
                                st.markdown("### Audio Summary")
                                audio_summary = summarize_text(audio_text, max_length=200, min_length=50, 
                                                            summarizer=summarization_model)
                                st.markdown(f'<div class="summary-box">{audio_summary}</div>', unsafe_allow_html=True)
                        
                        # Display a selection of frames
                        st.markdown("### Key Frames")
                        col1, col2, col3 = st.columns(3)
                        
                        if len(frames) >= 3:
                            # Display 3 frames: beginning, middle, end
                            indices = [0, len(frames)//2, len(frames)-1]
                            cols = [col1, col2, col3]
                            
                            for i, col in enumerate(cols):
                                with col:
                                    idx = indices[i]
                                    frame = frames[idx]
                                    st.image(frame, caption=f"Frame {idx}", use_column_width=True)
                        else:
                            # Display all frames if less than 3
                            cols = [col1, col2, col3]
                            for i, frame in enumerate(frames):
                                with cols[i % 3]:
                                    st.image(frame, caption=f"Frame {i}", use_column_width=True)
                        
                        # Option to download frames
                        st.markdown("### Download Options")
                        
                        # Create a zip of frames
                        if st.button("Download Frames as ZIP"):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
                                temp_zip_path = temp_zip.name
                            
                            import zipfile
                            with zipfile.ZipFile(temp_zip_path, 'w') as zipf:
                                for i, frame in enumerate(frames):
                                    # Convert numpy array to PIL Image
                                    img = Image.fromarray(frame)
                                    # Save to a temporary file
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                                        temp_img_path = temp_img.name
                                        img.save(temp_img_path)
                                    
                                    # Add to zip
                                    zipf.write(temp_img_path, f"frame_{i}.jpg")
                                    
                                    # Clean up
                                    os.remove(temp_img_path)
                            
                            # Provide download link
                            with open(temp_zip_path, "rb") as f:
                                bytes_data = f.read()
                                b64 = base64.b64encode(bytes_data).decode()
                                href = f'<a href="data:application/zip;base64,{b64}" download="video_frames.zip">Download ZIP</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            
                            # Clean up
                            os.remove(temp_zip_path)
                    else:
                        st.error("Failed to process video. Please try another file.")

    # --------------------- TAB 3: CHAT ANALYZER ---------------------
    with tab3:
        st.markdown('<h2 class="sub-header">Chat Conversation Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
        Upload exported chat conversations from WhatsApp, Slack, Teams, or other platforms.
        The analyzer will extract insights about the conversation dynamics.
        </div>
        """, unsafe_allow_html=True)
        
        # Upload chat file
        chat_file = st.file_uploader("Upload chat export:", type=['txt'])
        
        if chat_file:
            chat_text = chat_file.getvalue().decode("utf-8")
            
            # Process chat
            messages, chat_format = analyze_chat_data(chat_text)
            
            if messages:
                st.success(f"Chat data processed! Detected format: {chat_format.capitalize()}")
                
                # Show message statistics
                st.markdown("### Chat Statistics")
                
                # Basic metrics
                total_messages = len(messages)
                unique_senders = len(set(msg["sender"] for msg in messages))
                
                col_chat1, col_chat2, col_chat3 = st.columns(3)
                col_chat1.metric("Total Messages", total_messages)
                col_chat2.metric("Participants", unique_senders)
                
                # Calculate date range
                if messages[0]["timestamp"] and messages[-1]["timestamp"]:
                    time_span = messages[-1]["timestamp"] - messages[0]["timestamp"]
                    days = time_span.days + (time_span.seconds / 86400)
                    col_chat3.metric("Duration (days)", f"{days:.1f}")
                
                # Generate visualizations
                fig1, fig2, fig3 = visualize_chat_metrics(messages)
                
                if fig1 and fig2 and fig3:
                    st.pyplot(fig1)
                    st.pyplot(fig2)
                    st.pyplot(fig3)
                
                # Show conversation summary
                st.markdown("### Conversation Summary")
                
                # Compile all messages
                all_text = "\n".join([msg["message"] for msg in messages])
                
                # Generate summary
                if model_loaded and len(all_text) > 100:
                    chat_summary = summarize_text(all_text, max_length=200, min_length=50, 
                                                summarizer=summarization_model)
                    st.markdown(f'<div class="summary-box">{chat_summary}</div>', unsafe_allow_html=True)
                else:
                    st.info("Chat content too short for meaningful summary.")
                
                # Show key topics/keywords
                if nlp:
                    st.markdown("### Key Topics")
                    keywords = extract_keywords(all_text, nlp, top_n=15)
                    
                    if keywords:
                        # Create horizontal bar chart for top keywords
                        fig, ax = plt.subplots(figsize=(10, 6))
                        keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Count"])
                        sns.barplot(x="Count", y="Keyword", data=keyword_df, ax=ax)
                        ax.set_title("Top Keywords in Conversation")
                        st.pyplot(fig)
                    
                    # Sentiment analysis by participant
                    st.markdown("### Sentiment by Participant")
                    
                    # Group messages by sender
                    sender_texts = defaultdict(list)
                    for msg in messages:
                        sender_texts[msg["sender"]].append(msg["message"])
                    
                    # Calculate sentiment for each sender
                    sender_sentiments = {}
                    for sender, texts in sender_texts.items():
                        combined_text = " ".join(texts)
                        sentiment, polarity = analyze_sentiment(combined_text, nlp)
                        sender_sentiments[sender] = polarity
                    
                    # Create dataframe and visualize
                    sentiment_df = pd.DataFrame({
                        'Sender': list(sender_sentiments.keys()),
                        'Sentiment': list(sender_sentiments.values())
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#4CAF50' if s > 0 else '#F44336' if s < 0 else '#9E9E9E' for s in sentiment_df['Sentiment']]
                    sns.barplot(x='Sentiment', y='Sender', data=sentiment_df, palette=colors, ax=ax)
                    ax.set_title("Sentiment Analysis by Participant")
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    st.pyplot(fig)
                
                # Interactive message browser
                st.markdown("### Message Browser")
                
                message_df = pd.DataFrame([
                    {
                        "Time": msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if msg["timestamp"] else "Unknown",
                        "Sender": msg["sender"],
                        "Message": msg["message"][:100] + ("..." if len(msg["message"]) > 100 else "")
                    }
                    for msg in messages
                ])
                
                st.dataframe(message_df, height=300)
                
                # Allow filtering by participant
                st.markdown("### Filter Messages")
                
                selected_sender = st.selectbox(
                    "Select participant:",
                    options=["All"] + list(set(msg["sender"] for msg in messages))
                )
                
                if selected_sender != "All":
                    filtered_messages = [msg for msg in messages if msg["sender"] == selected_sender]
                else:
                    filtered_messages = messages
                
                # Show filtered messages
                for msg in filtered_messages[:10]:  # Show first 10 messages
                    st.markdown(
                        f"""<div class="chat-message user-message">
                        <b>{msg["sender"]}</b> ({msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if msg["timestamp"] else "Unknown"})<br>
                        {msg["message"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                
                if len(filtered_messages) > 10:
                    st.info(f"{len(filtered_messages) - 10} more messages not shown...")
            else:
                st.error("Failed to parse chat data. Please try another file or format.")
    
    # --------------------- TAB 4: REAL-TIME SESSION ---------------------
    with tab4:
        st.markdown('<h2 class="sub-header">Real-Time Summarization Session</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-text">
        Use this space for real-time note-taking and automatic summarization.
        Perfect for meetings, lectures, or any situation where you need ongoing summaries.
        </div>
        """, unsafe_allow_html=True)
        
        # Session control
        session_col1, session_col2 = st.columns([3, 1])
        
        with session_col1:
            session_name = st.text_input("Session Name:", value=f"Session {datetime.now().strftime('%Y-%m-%d')}")
        
        with session_col2:
            if st.session_state.realtime_summarizer.is_session_active():
                if st.button("End Session"):
                    st.session_state.realtime_summarizer.end_session()
                    st.success("Session ended. Final summary available below.")
                    st.experimental_rerun()
            else:
                if st.button("Start New Session"):
                    st.session_state.realtime_summarizer.start_session(session_name)
                    st.success(f"Session '{session_name}' started!")
                    st.experimental_rerun()
        
        # Real-time input area
        if st.session_state.realtime_summarizer.is_session_active():
            # Input area
            new_entry = st.text_area("Add notes (press Enter to submit):", key="realtime_notes", height=100)
            
            if st.button("Add to Session"):
                if new_entry.strip():
                    st.session_state.realtime_summarizer.add_message(new_entry)

                    st.success("Note added!")
                    # Clear the input
                    st.session_state.realtime_notes = ""
                    st.experimental_rerun()
                else:
                    st.warning("Please enter some text before adding.")
            
            # Display current session content
            st.markdown("### Session Content")
            
            entries = st.session_state.realtime_summarizer.get_entries()
            
from datetime import datetime

entries = st.session_state.realtime_summarizer.get_entries() if st.session_state.realtime_summarizer.is_session_active() else []
for i, entry in enumerate(entries):
    # Ensure timestamp is a datetime object
    if isinstance(entry['timestamp'], str):
        entry_time = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S")  # or adjust format as needed
    else:
        entry_time = entry['timestamp']

    st.markdown(
        f"""<div class="chat-message user-message">
        <b>Entry {i+1}</b> ({entry_time.strftime("%H:%M:%S")})<br>
        {entry['text']}
        </div>""",
        unsafe_allow_html=True
    )


            
            # Display real-time summary
    st.markdown("### Current Summary")
            
    if entries:
                # Generate summary
                if 'summarization_model' not in locals():
                    summarization_model, model_loaded = load_summarization_model()
                current_summary = st.session_state.realtime_summarizer.get_current_summary(summarizer=summarization_model)
                
                st.markdown(
                    f"""<div class="realtime-update">
                    <b>Last updated:</b> {datetime.now().strftime("%H:%M:%S")}<br><br>
                    {current_summary}
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Key points extraction
                if 'nlp' not in locals() or nlp is None:
                    nlp = load_spacy_model() if check_spacy_model() else None
                key_points = st.session_state.realtime_summarizer.extract_key_points(nlp)
                
                if key_points:
                    st.markdown("### Key Points")
                    for point in key_points:
                        st.markdown(f"- {point}")
    else:
                st.info("Add some notes to generate a summary.")
else:
            st.info("Start a new session to begin real-time summarization.")
            
            # Show previous sessions if any
            previous_sessions = st.session_state.summary_explorer.get_saved_sessions()
            
            if previous_sessions:
                st.markdown("### Previous Sessions")
                
                selected_session = st.selectbox(
                    "Select a previous session to view:",
                    options=previous_sessions
                )
                
                if selected_session:
                    session_data = st.session_state.summary_explorer.load_session(selected_session)
                    
                    if session_data:
                        st.markdown(f"### {selected_session}")
                        
                        st.markdown("#### Final Summary")
                        st.markdown(f'<div class="summary-box">{session_data["summary"]}</div>', unsafe_allow_html=True)
                        
                        st.markdown("#### Session Content")
                        for i, entry in enumerate(session_data["entries"]):
                            st.markdown(
                                f"""<div class="chat-message user-message">
                                <b>Entry {i+1}</b> ({entry['timestamp']})<br>
                                {entry['text']}
                                </div>""",
                                unsafe_allow_html=True
                            )
            else:
                st.info("No previous sessions found.")
        
        # Add export options
entries = st.session_state.realtime_summarizer.get_entries() if st.session_state.realtime_summarizer.is_session_active() else []
if st.session_state.realtime_summarizer.is_session_active() and entries:
            st.markdown("### Export Options")
            
            export_format = st.radio("Choose export format:", ["Text", "Markdown", "PDF"], horizontal=True)
            
            if st.button("Export Summary"):
                summary_text = st.session_state.realtime_summarizer.get_current_summary(summarizer=summarization_model)
                session_name = st.session_state.realtime_summarizer.get_session_name()
                
                if export_format == "Text":
                    # Create text file
                    text_content = f"Session: {session_name}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\nSUMMARY:\n{summary_text}\n\nDETAILED NOTES:\n"
                    
                    for i, entry in enumerate(entries):
                        text_content += f"\nEntry {i+1} ({entry['timestamp'].strftime('%H:%M:%S')}):\n{entry['text']}\n"
                    
                    # Create download link
                    b64 = base64.b64encode(text_content.encode()).decode()
                    href = f'<a href="data:file/txt;base64,{b64}" download="{session_name}.txt">Download Text File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                elif export_format == "Markdown":
                    # Create markdown file
                    md_content = f"# Session: {session_name}\nDate: {datetime.now().strftime('%Y-%m-%d')}\n\n## SUMMARY:\n{summary_text}\n\n## DETAILED NOTES:\n"
                    
                    for i, entry in enumerate(entries):
                        md_content += f"\n### Entry {i+1} ({entry['timestamp'].strftime('%H:%M:%S')}):\n{entry['text']}\n"
                    
                    # Create download link
                    b64 = base64.b64encode(md_content.encode()).decode()
                    href = f'<a href="data:file/md;base64,{b64}" download="{session_name}.md">Download Markdown File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                elif export_format == "PDF":
                    try:
                        from reportlab.lib.pagesizes import letter
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                        from reportlab.lib.styles import getSampleStyleSheet
                        import io
                        
                        # Create PDF
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=letter)
                        styles = getSampleStyleSheet()
                        
                        # Content
                        content = []
                        
                        # Title
                        content.append(Paragraph(f"Session: {session_name}", styles['Title']))
                        content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
                        content.append(Spacer(1, 12))
                        
                        # Summary
                        content.append(Paragraph("SUMMARY:", styles['Heading2']))
                        content.append(Paragraph(summary_text, styles['Normal']))
                        content.append(Spacer(1, 12))
                        
                        # Detailed Notes
                        content.append(Paragraph("DETAILED NOTES:", styles['Heading2']))
                        
                        for i, entry in enumerate(entries):
                            content.append(Spacer(1, 12))
                            content.append(Paragraph(f"Entry {i+1} ({entry['timestamp'].strftime('%H:%M:%S')})", styles['Heading3']))
                            content.append(Paragraph(entry['text'], styles['Normal']))
                        
                        # Build PDF
                        doc.build(content)
                        
                        # Create download link
                        b64 = base64.b64encode(buffer.getvalue()).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="{session_name}.pdf">Download PDF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error creating PDF: {e}")
                        st.info("Please install ReportLab for PDF export: pip install reportlab")

if __name__ == "__main__":
    main()