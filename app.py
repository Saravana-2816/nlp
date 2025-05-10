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

st.set_page_config(
    page_title="Text, Video & Chat Summarizer",
    page_icon="📊",
    layout="wide"
)
# Try importing spacy, but provide fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    st.error("spaCy is not installed. Please install it using: pip install spacy")
    st.info("Some features like entity extraction will be limited.")

# Try importing pydub for audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    st.error("pydub is not installed. Please install it using: pip install pydub")
    st.info("Audio extraction features will be limited.")

# Try importing pytube for YouTube downloads
try:
    import pytube
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False
    st.error("pytube is not installed. Please install it using: pip install pytube")
    st.info("YouTube download features will be disabled.")

# Add custom CSS
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
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading summarization model: {str(e)}")
        return None, None

@st.cache_resource
def load_spacy_model():
    if SPACY_AVAILABLE:
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy model 'en_core_web_sm' not found. Installing it now...")
            try:
                # Try to download the model programmatically
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                                check=True, capture_output=True)
                return spacy.load("en_core_web_sm")
            except Exception as e:
                st.error(f"Failed to install spaCy model: {str(e)}")
                return None
    return None

# Text summarization function
def summarize_text(text, max_length=150, min_length=50):
    tokenizer, model = load_summarization_model()
    if not tokenizer or not model:
        return "Error loading summarization model."
    
    try:
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        
        # Handle longer texts by chunking if needed
        if len(text.split()) > 1024:
            chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=max_length//len(chunks), 
                                    min_length=min_length//len(chunks), 
                                    do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Extract key entities and generate insights
def extract_entities(text):
    if not SPACY_AVAILABLE:
        return {"NOTE": ["Entity extraction requires spaCy"]}
    
    nlp = load_spacy_model()
    if not nlp:
        return {"ERROR": ["Could not load spaCy model"]}
    
    try:
        doc = nlp(text)
        
        entities = {}
        for ent in doc.ents:
            ent_type = ent.label_
            if ent_type not in entities:
                entities[ent_type] = []
            if ent.text not in entities[ent_type]:
                entities[ent_type].append(ent.text)
        
        return entities
    except Exception as e:
        return {"ERROR": [f"Error extracting entities: {str(e)}"]}

# Simple tokenization fallback when spaCy isn't available
def simple_tokenize(text):
    # Remove punctuation and convert to lowercase
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Split on whitespace
    return text.split()

# Generate word cloud from text
def generate_wordcloud(text):
    try:
        if SPACY_AVAILABLE:
            nlp = load_spacy_model()
            if nlp:
                doc = nlp(text)
                # Filter out stopwords and punctuation
                filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
                processed_text = " ".join(filtered_tokens)
            else:
                # Fallback to simple tokenization
                tokens = simple_tokenize(text)
                processed_text = " ".join(tokens)
        else:
            # Fallback to simple tokenization
            tokens = simple_tokenize(text)
            processed_text = " ".join(tokens)
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             colormap='viridis', max_words=100).generate(processed_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        # Return an empty figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error generating word cloud: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

# Function to extract audio from video using pydub
def extract_audio_from_video(video_path):
    if not PYDUB_AVAILABLE:
        st.error("Audio extraction requires pydub. Please install it using: pip install pydub")
        return None
        
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    # Use ffmpeg via pydub to extract audio
    try:
        video = AudioSegment.from_file(video_path)
        video.export(temp_audio_path, format="wav")
        return temp_audio_path
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return None

# Improved speech-to-text with noise reduction and improved accuracy
def speech_to_text(audio_path):
    if not audio_path:
        return "[Error extracting audio]"
        
    recognizer = sr.Recognizer()
    
    try:
        # Load audio file
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(audio_path)
            
            # Apply basic noise reduction (low-pass filter)
            # This helps improve recognition accuracy
            filtered_audio = audio.low_pass_filter(2000)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_filtered:
                filtered_path = temp_filtered.name
                filtered_audio.export(filtered_path, format="wav")
            
            duration_seconds = len(audio) / 1000  # pydub uses milliseconds
            
            # Process in smaller chunks (20-second) for better accuracy
            chunk_size = 20  # in seconds
            full_text = ""
            
            # Add progress indicator for long audio files
            if duration_seconds > 60:
                progress_bar = st.progress(0)
            
            for i in range(0, int(duration_seconds), chunk_size):
                if duration_seconds > 60:
                    progress_bar.progress(i / duration_seconds)
                    
                end_time = min(i + chunk_size, duration_seconds)
                
                # Extract chunk (in milliseconds for pydub)
                chunk = filtered_audio[i*1000:(end_time*1000)]
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                    chunk_path = temp_chunk.name
                    chunk.export(chunk_path, format="wav")
                
                # Process chunk with multiple attempts for better accuracy
                with sr.AudioFile(chunk_path) as source:
                    # Adjust for ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=min(0.5, end_time-i))
                    audio_data = recognizer.record(source)
                    
                    # Try multiple recognition services for better results
                    try:
                        # First try Google (usually more accurate but requires internet)
                        text = recognizer.recognize_google(audio_data)
                    except sr.UnknownValueError:
                        try:
                            # Fall back to Sphinx (offline but less accurate)
                            text = recognizer.recognize_sphinx(audio_data)
                        except:
                            text = "[inaudible section]"
                    except Exception as e:
                        text = f"[Error: {str(e)}]"
                
                full_text += " " + text
                
                # Clean up chunk file
                os.unlink(chunk_path)
            
            # Clean up filtered audio file
            os.unlink(filtered_path)
            
            if duration_seconds > 60:
                progress_bar.empty()
        else:
            # Fallback to processing the entire file at once
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    full_text = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    full_text = "[Speech recognition could not understand audio]"
                except Exception as e:
                    full_text = f"[Error processing audio: {str(e)}]"
        
        return full_text.strip()
    except Exception as e:
        return f"[Error processing audio: {str(e)}]"

# Improved frame extraction with scene detection
def extract_video_frames(video_path, num_frames=10):
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frames = []
        timestamps = []
        
        # Try scene detection for more meaningful frames
        try:
            # Calculate frame differences to detect scene changes
            prev_frame = None
            scene_changes = []
            
            # Sample frames for scene detection
            sample_rate = max(1, total_frames // 100)
            
            for i in range(0, total_frames, sample_rate):
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                if not ret:
                    break
                
                # Convert to grayscale for difference calculation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_frame is not None:
                    # Calculate absolute difference between current and previous frame
                    frame_diff = cv2.absdiff(gray, prev_frame)
                    # Calculate mean difference
                    mean_diff = np.mean(frame_diff)
                    scene_changes.append((i, mean_diff))
                
                prev_frame = gray
            
            # Sort by difference magnitude to find biggest scene changes
            scene_changes.sort(key=lambda x: x[1], reverse=True)
            
            # Take top scene changes
            top_scenes = scene_changes[:num_frames]
            # Sort frames by their position in video
            top_scenes.sort(key=lambda x: x[0])
            
            # If we found enough scene changes, use them
            if len(top_scenes) >= num_frames // 2:
                for frame_idx, _ in top_scenes[:num_frames]:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = video.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        timestamp = frame_idx / fps
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        timestamps.append(f"{minutes:02d}:{seconds:02d}")
                
        except Exception as e:
            # Fallback to regular interval extraction
            st.warning(f"Scene detection failed, using regular intervals: {str(e)}")
            
            # Extract frames at regular intervals if scene detection fails
            if not frames:
                interval = max(1, total_frames // num_frames)
                for i in range(min(num_frames, total_frames)):
                    frame_idx = i * interval
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = video.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        timestamp = frame_idx / fps
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        timestamps.append(f"{minutes:02d}:{seconds:02d}")
        
        video.release()
        return frames, timestamps
    except Exception as e:
        st.error(f"Error extracting video frames: {str(e)}")
        return [], []

# Generate video summary metrics
def generate_video_metrics(video_path):
    try:
        video = cv2.VideoCapture(video_path)
        
        # Basic video metrics
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Enhanced metrics
        # Sample frames to calculate average brightness and motion
        sample_frames = min(100, frame_count)
        brightness_values = []
        motion_values = []
        prev_frame = None
        
        for i in range(0, frame_count, max(1, frame_count // sample_frames)):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                break
                
            # Calculate brightness
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:,:,2])
            brightness_values.append(brightness)
            
            # Calculate motion if possible
            if prev_frame is not None:
                # Convert to grayscale
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference
                diff = cv2.absdiff(gray1, gray2)
                motion = np.mean(diff)
                motion_values.append(motion)
                
            prev_frame = frame
        
        # Calculate average values
        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        avg_motion = np.mean(motion_values) if motion_values else 0
        
        video.release()
        
        return {
            "Duration": f"{int(duration // 60)} min {int(duration % 60)} sec",
            "Resolution": f"{width}x{height}",
            "FPS": f"{fps:.2f}",
            "Total Frames": frame_count,
            "Avg Brightness": f"{avg_brightness:.1f}/255",
            "Motion Level": f"{'High' if avg_motion > 15 else 'Medium' if avg_motion > 5 else 'Low'}"
        }
    except Exception as e:
        st.error(f"Error generating video metrics: {str(e)}")
        return {
            "Duration": "Unknown",
            "Resolution": "Unknown",
            "FPS": "Unknown",
            "Total Frames": "Unknown",
            "Avg Brightness": "Unknown",
            "Motion Level": "Unknown"
        }

# Display installation instructions
def show_installation_instructions():
    with st.expander("Installation Instructions"):
        st.markdown("""
        ### Required Packages
        
        ```
        pip install streamlit numpy pandas matplotlib pytube opencv-python pillow torch transformers spacy wordcloud SpeechRecognition pydub seaborn
        ```
        
        ### Install spaCy Language Model
        
        ```
        python -m spacy download en_core_web_sm
        ```
        
        ### Install FFmpeg (for audio processing)
        
        **Ubuntu/Debian:**
        ```
        sudo apt-get install ffmpeg
        ```
        
        **macOS:**
        ```
        brew install ffmpeg
        ```
        
        **Windows:**
        Download from the [FFmpeg website](https://ffmpeg.org/download.html) and add to your PATH
        """)

# NEW FEATURE: Group Chat Summarizer functions

# Parse chat messages
def parse_chat_messages(chat_text):
    """
    Parse chat messages in the format "Name: Message" and return structured data.
    """
    try:
        # Regular expression to match "Name: Message" pattern
        pattern = r"(.+?)\s*:\s*(.+)"
        
        messages = []
        current_speaker = None
        current_message = []
        
        for line in chat_text.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            match = re.match(pattern, line)
            if match:
                # If we were building a message for a previous speaker, save it
                if current_speaker and current_message:
                    messages.append({
                        "speaker": current_speaker,
                        "message": " ".join(current_message)
                    })
                    current_message = []
                
                current_speaker = match.group(1).strip()
                current_message = [match.group(2).strip()]
            else:
                # Continuation of the previous message
                if current_speaker:
                    current_message.append(line)
                else:
                    # If no current speaker, treat this as a system message or invalid format
                    current_speaker = "System"
                    current_message = [line]
        
        # Add the last message if there is one
        if current_speaker and current_message:
            messages.append({
                "speaker": current_speaker,
                "message": " ".join(current_message)
            })
        
        return messages
    except Exception as e:
        st.error(f"Error parsing chat messages: {str(e)}")
        return []

# Analyze chat engagement
def analyze_chat_engagement(messages):
    """
    Analyze chat engagement metrics by speaker.
    """
    speakers = {}
    
    # Count messages and calculate message lengths
    for msg in messages:
        speaker = msg["speaker"]
        message = msg["message"]
        
        if speaker not in speakers:
            speakers[speaker] = {
                "message_count": 0,
                "total_words": 0,
                "messages": []
            }
        
        speakers[speaker]["message_count"] += 1
        speakers[speaker]["total_words"] += len(message.split())
        speakers[speaker]["messages"].append(message)
    
    # Calculate engagement metrics
    results = []
    for speaker, data in speakers.items():
        avg_words = data["total_words"] / data["message_count"] if data["message_count"] > 0 else 0
        
        results.append({
            "speaker": speaker,
            "message_count": data["message_count"],
            "total_words": data["total_words"],
            "avg_words": avg_words,
            "participation_pct": 0  # Will calculate after
        })
    
    # Calculate participation percentages
    total_messages = sum(r["message_count"] for r in results)
    if total_messages > 0:
        for i in range(len(results)):
            results[i]["participation_pct"] = (results[i]["message_count"] / total_messages) * 100
    
    # Sort by message count
    results.sort(key=lambda x: x["message_count"], reverse=True)
    
    return results

# Identify topic clusters
def identify_topics(messages, num_topics=3):
    """
    Identify main topics or themes in the chat.
    Uses simple keyword frequency if spaCy is not available.
    """
    if not SPACY_AVAILABLE:
        # Simple fallback using keyword frequency
        word_counts = Counter()
        
        for msg in messages:
            # Simple tokenization
            words = simple_tokenize(msg["message"])
            word_counts.update(words)
        
        # Filter out common stopwords
        common_stopwords = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", 
            "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
            "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
            "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
            "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", 
            "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
            "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
            "of", "at", "by", "for", "with", "about", "against", "between", "into", 
            "through", "during", "before", "after", "above", "below", "to", "from", 
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", 
            "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
            "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", 
            "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
            "will", "just", "don", "should", "now"
        }
        
        for word in list(word_counts.keys()):
            if word in common_stopwords or len(word) < 3:
                del word_counts[word]
        
        # Get the most common topics
        topics = [word for word, count in word_counts.most_common(num_topics)]
        return topics
    else:
        # Use spaCy for better topic extraction
        nlp = load_spacy_model()
        if not nlp:
            return ["Topic analysis requires spaCy model"]
            
        # Combine all messages into one text
        all_text = " ".join(msg["message"] for msg in messages)
        
        # Process with spaCy
        doc = nlp(all_text)
        
        # Extract noun phrases and named entities as potential topics
        topics = []
        seen = set()
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC", "GPE"]:
                if ent.text.lower() not in seen and len(ent.text) > 3:
                    topics.append(ent.text)
                    seen.add(ent.text.lower())
        
        # Add noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in seen and len(chunk.text) > 3:
                # Filter out chunks that are just pronouns or determiners
                if not all(token.pos_ in ["PRON", "DET"] for token in chunk):
                    topics.append(chunk.text)
                    seen.add(chunk.text.lower())
        
        # If we don't have enough topics, add frequently occurring nouns
        if len(topics) < num_topics:
            # Count noun frequencies
            noun_counts = Counter()
            for token in doc:
                if token.pos_ == "NOUN" and not token.is_stop:
                    noun_counts[token.text] += 1
            
            # Add top nouns that aren't already in topics
            for noun, _ in noun_counts.most_common(num_topics * 2):
                if noun.lower() not in seen:
                    topics.append(noun)
                    seen.add(noun.lower())
                    if len(topics) >= num_topics:
                        break
        
        return topics[:num_topics]

# Summarize chat conversation
def summarize_chat(messages, engagement_data, topics):
    """
    Generate a comprehensive summary of the chat conversation.
    """
    # Create a simplified text representation of the conversation
    conversation_text = ""
    for msg in messages:
        conversation_text += f"{msg['message']}\n\n"
    
    # Use the general summarization function
    summary = summarize_text(conversation_text, max_length=200, min_length=100)
    
    # Create a more structured summary
    top_participants = [e["speaker"] for e in engagement_data[:3]] if len(engagement_data) > 0 else []
    
    structured_summary = {
        "general_summary": summary,
        "main_topics": topics,
        "top_participants": top_participants,
        "speaker_count": len(engagement_data),
        "message_count": sum(e["message_count"] for e in engagement_data),
        "most_active": engagement_data[0]["speaker"] if engagement_data else "Unknown"
    }
    
    return structured_summary

# Generate visualization of speaker participation
def visualize_chat_participation(engagement_data):
    """
    Create a visualization of chat participation.
    """
    speakers = [e["speaker"] for e in engagement_data]
    message_counts = [e["message_count"] for e in engagement_data]
    word_counts = [e["total_words"] for e in engagement_data]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Speaker": speakers,
        "Messages": message_counts,
        "Words": word_counts
    })
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Message count bar chart
    sns.barplot(x="Messages", y="Speaker", data=df, ax=ax1, palette="viridis")
    ax1.set_title("Message Count by Speaker")
    ax1.set_xlabel("Number of Messages")
    ax1.set_ylabel("")
    
    # Word count bar chart
    sns.barplot(x="Words", y="Speaker", data=df, ax=ax2, palette="viridis")
    ax2.set_title("Word Count by Speaker")
    ax2.set_xlabel("Number of Words")
    ax2.set_ylabel("")
    
    plt.tight_layout()
    return fig

# Find conversation threads
# Find conversation threads
def identify_conversation_threads(messages):
    """
    Identify conversation threads between speakers.
    """
    threads = []
    
    # Look for direct replies (simple heuristic based on consecutive messages)
    for i in range(1, len(messages)):
        prev_msg = messages[i-1]
        curr_msg = messages[i]
        
        # Different speakers in consecutive messages suggest a reply
        if prev_msg["speaker"] != curr_msg["speaker"]:
            # Check if the current message might be addressing the previous speaker
            # Simple heuristic: contains their name or common reply patterns
            prev_speaker_name = prev_msg["speaker"].split()[0]  # First name
            curr_message = curr_msg["message"].lower()
            
            if (prev_speaker_name.lower() in curr_message or 
                curr_message.startswith(("yes", "no", "agree", "disagree", "thanks", "thank", "ok", "okay"))):
                threads.append({
                    "speaker1": prev_msg["speaker"],
                    "message1": prev_msg["message"],
                    "speaker2": curr_msg["speaker"],
                    "message2": curr_msg["message"]
                })
    
    return threads

# Main application
def main():
    st.markdown('<h1 class="main-header">📊 Text, Video & Chat Summarization Tool</h1>', unsafe_allow_html=True)
    
    st.markdown('<p class="info-text">This tool analyzes and summarizes text, video content, and group chat conversations, extracting key information and visualizing insights.</p>', unsafe_allow_html=True)
    
    # Show installation button
    if st.button("⚙️ Show Installation Instructions"):
        show_installation_instructions()
    
    # Check for spaCy model at startup
    spacy_ready = check_spacy_model() if SPACY_AVAILABLE else False
    
    # Create tabs for different input types
    tab1, tab2, tab3 = st.tabs(["Text Summarization", "Video Summarization", "Group Chat Summarization"])
    
    # Text Summarization Tab
    with tab1:
        st.markdown('<h2 class="sub-header">Text Summarization</h2>', unsafe_allow_html=True)
        
        text_input_option = st.radio("Input Method:", ["Enter Text", "Upload Text File"])
        
        if text_input_option == "Enter Text":
            text_input = st.text_area("Enter the text to summarize:", height=250)
        else:
            uploaded_file = st.file_uploader("Upload a text file:", type=["txt", "md", "pdf"])
            text_input = ""
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".pdf"):
                    st.warning("PDF support is limited to text extraction only.")
                    # Note: For production, add PDF text extraction code here
                    text_input = "PDF text would be extracted here."
                else:
                    text_input = uploaded_file.read().decode("utf-8")
        
        col1, col2 = st.columns(2)
        with col1:
            min_length = st.slider("Minimum summary length:", 30, 200, 50)
        with col2:
            max_length = st.slider("Maximum summary length:", 100, 500, 150)
        
        if st.button("Summarize Text") and text_input:
            with st.spinner("Analyzing text..."):
                progress_bar = st.progress(0)
                
                # Step 1: Generate summary
                progress_bar.progress(30)
                summary = summarize_text(text_input, max_length=max_length, min_length=min_length)
                
                # Step 2: Extract entities and insights
                progress_bar.progress(60)
                entities = extract_entities(text_input)
                
                # Step 3: Generate visualizations
                progress_bar.progress(90)
                wordcloud_fig = generate_wordcloud(text_input)
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
            
            # Display results
            st.markdown("### Summary")
            st.write(summary)
            
            st.markdown("### Key Entities")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    with st.expander(f"{entity_type} ({len(entity_list)})"):
                        st.write(", ".join(entity_list))
            
            st.markdown("### Word Cloud")
            st.pyplot(wordcloud_fig)
            
            # Text statistics
            st.markdown("### Text Statistics")
            word_count = len(text_input.split())
            sentence_count = len(text_input.split('.'))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Word Count", word_count)
            col2.metric("Sentence Count", sentence_count)
            col3.metric("Compression Ratio", f"{len(summary.split()) / word_count:.1%}")
    
    # Video Summarization Tab
    with tab2:
        st.markdown('<h2 class="sub-header">Video Summarization</h2>', unsafe_allow_html=True)
        
        video_input_option = st.radio("Video Source:", ["Upload Video", "YouTube URL"])
        
        video_path = None
        if video_input_option == "Upload Video":
            uploaded_video = st.file_uploader("Upload a video file:", type=["mp4", "mov", "avi", "mkv"])
            if uploaded_video is not None:
                # Save uploaded video to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    video_path = tmp_file.name
        else:
            if not PYTUBE_AVAILABLE:
                st.error("YouTube functionality requires pytube. Please install it using: pip install pytube")
            else:
                youtube_url = st.text_input("Enter YouTube URL:")
                if youtube_url:
                    try:
                        with st.spinner("Downloading YouTube video..."):
                            yt = pytube.YouTube(youtube_url)
                            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                            
                            # Save to temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                                video_path = tmp_file.name
                            
                            stream.download(filename=video_path)
                            st.success("Video downloaded successfully!")
                    except Exception as e:
                        st.error(f"Error downloading YouTube video: {str(e)}")
        
        # Add video processing options
        with st.expander("Advanced Video Processing Options"):
            col1, col2 = st.columns(2)
            with col1:
                frame_count = st.slider("Number of key frames to extract:", 5, 20, 10)
            with col2:
                audio_quality = st.select_slider(
                    "Speech recognition quality:",
                    options=["Low (Fast)", "Medium", "High (Slow)"],
                    value="Medium"
                )
        
        if video_path and st.button("Analyze Video"):
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                
                # Step 1: Extract audio and convert to text
                progress_bar.progress(20)
                audio_path = extract_audio_from_video(video_path)
                progress_bar.progress(40)
                
                extracted_text = speech_to_text(audio_path) if audio_path else "[Error extracting audio]"
                progress_bar.progress(60)
                
                # Step 2: Summarize the extracted text
                if extracted_text and extracted_text != "[Error extracting audio]":
                    text_summary = summarize_text(extracted_text)
                else:
                    text_summary = "Could not generate summary due to audio extraction issues."
                progress_bar.progress(70)
                
                # Step 3: Extract video frames for visualization
                frames, timestamps = extract_video_frames(video_path, num_frames=frame_count)
                progress_bar.progress(80)
                
                # Step 4: Generate video metrics
                video_metrics = generate_video_metrics(video_path)
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                
                # Cleanup temp files
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            
            # Display video preview
            st.markdown("### Video Preview")
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Display video summary and transcription
            st.markdown("### Video Summary")
            st.write(text_summary)
            
            with st.expander("Full Transcription"):
                st.write(extracted_text)
            
            # Display key frames
            if frames:
                st.markdown("### Key Frames")
                # Create rows of 5 frames each
                for i in range(0, len(frames), 5):
                    cols = st.columns(5)
                    for j, (frame, timestamp) in enumerate(zip(frames[i:i+5], timestamps[i:i+5])):
                        with cols[j]:
                            st.image(frame, caption=f"Time: {timestamp}")
            
            # Display video metrics - enhanced with more details
            st.markdown("### Video Metrics")
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Duration", video_metrics["Duration"])
                st.metric("Resolution", video_metrics["Resolution"])
            with metrics_cols[1]:
                st.metric("FPS", video_metrics["FPS"])
                st.metric("Total Frames", video_metrics["Total Frames"])
            with metrics_cols[2]:
                st.metric("Average Brightness", video_metrics["Avg Brightness"])
                st.metric("Motion Level", video_metrics["Motion Level"])
            
            # Generate and display word cloud from transcription
            if extracted_text and extracted_text != "[Error extracting audio]":
                st.markdown("### Content Word Cloud")
                wordcloud_fig = generate_wordcloud(extracted_text)
                st.pyplot(wordcloud_fig)
                
                # Extract and display entities if spaCy is available
                if SPACY_AVAILABLE:
                    entities = extract_entities(extracted_text)
                    st.markdown("### Mentioned Entities")
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            with st.expander(f"{entity_type} ({len(entity_list)})"):
                                st.write(", ".join(entity_list))
            
            # Cleanup the temporary video file
            video_file.close()
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    # NEW TAB: Group Chat Summarization
    with tab3:
        st.markdown('<h2 class="sub-header">Group Chat Summarization</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <p class="info-text">This feature analyzes group chat conversations, identifies key participants, topics, 
        and generates a summary of the discussion. Format should be "Name: Message" for each message.</p>
        """, unsafe_allow_html=True)
        
        chat_input_option = st.radio("Chat Input Method:", ["Enter Chat Transcript", "Upload Chat File"])
        
        if chat_input_option == "Enter Chat Transcript":
            chat_input = st.text_area("Enter the chat transcript (Format: 'Name: Message'):", 
                                     height=300,
                                     placeholder="Rishi: Hello everyone!\nSaravana: Hi Rishi, how are you?\nPriya: I wanted to discuss the project timeline...")
        else:
            uploaded_chat = st.file_uploader("Upload a chat transcript file:", type=["txt"])
            chat_input = ""
            if uploaded_chat is not None:
                chat_input = uploaded_chat.read().decode("utf-8")
        
        if st.button("Analyze Chat") and chat_input:
            with st.spinner("Analyzing chat conversation..."):
                progress_bar = st.progress(0)
                
                # Step 1: Parse the chat messages
                progress_bar.progress(20)
                parsed_messages = parse_chat_messages(chat_input)
                
                if not parsed_messages:
                    st.error("Could not parse any messages. Make sure to use the format 'Name: Message'")
                else:
                    # Step 2: Analyze engagement metrics
                    progress_bar.progress(40)
                    engagement_data = analyze_chat_engagement(parsed_messages)
                    
                    # Step 3: Identify topics
                    progress_bar.progress(60)
                    topics = identify_topics(parsed_messages)
                    
                    # Step 4: Generate summary
                    progress_bar.progress(80)
                    summary = summarize_chat(parsed_messages, engagement_data, topics)
                    
                    # Step 5: Identify conversation threads
                    threads = identify_conversation_threads(parsed_messages)
                    
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    # Display the results
                    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                    st.markdown("### Chat Summary")
                    st.write(summary["general_summary"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Metrics overview
                    st.markdown("### Conversation Overview")
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Participants", summary["speaker_count"])
                    metrics_cols[1].metric("Messages", summary["message_count"])
                    metrics_cols[2].metric("Main Speaker", summary["most_active"])
                    metrics_cols[3].metric("Topics Detected", len(summary["main_topics"]))
                    
                    # Display main topics
                    st.markdown("### Main Topics Discussed")
                    if topics:
                        for topic in topics:
                            st.markdown(f"- **{topic}**")
                    else:
                        st.info("No clear topics were identified.")
                    
                    # Display visualization of participation
                    st.markdown("### Participant Engagement")
                    participation_fig = visualize_chat_participation(engagement_data)
                    st.pyplot(participation_fig)
                    
                    # Display message details by speaker
                    st.markdown("### Message Breakdown by Participant")
                    for speaker_data in engagement_data:
                        with st.expander(f"{speaker_data['speaker']} ({speaker_data['message_count']} messages)"):
                            st.metric("Words", speaker_data["total_words"])
                            st.metric("Average Words per Message", f"{speaker_data['avg_words']:.1f}")
                            st.metric("Participation", f"{speaker_data['participation_pct']:.1f}%")
                    
                    # Display conversation threads
                    if threads:
                        st.markdown("### Key Conversation Threads")
                        for i, thread in enumerate(threads[:5]):  # Show up to 5 threads
                            with st.expander(f"Thread {i+1}: {thread['speaker1']} ↔ {thread['speaker2']}"):
                                st.markdown(f"**{thread['speaker1']}:** {thread['message1']}")
                                st.markdown(f"**{thread['speaker2']}:** {thread['message2']}")
                    
                    # Word cloud visualization
                    st.markdown("### Word Cloud from Conversation")
                    all_text = " ".join(msg["message"] for msg in parsed_messages)
                    wordcloud_fig = generate_wordcloud(all_text)
                    st.pyplot(wordcloud_fig)

# Add system module for subprocess calls
import sys

if __name__ == "__main__":
    main()