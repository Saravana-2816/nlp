import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import json
import re
from transformers import pipeline

class RealTimeSummarizer:
    """
    Handles real-time summarization of ongoing sessions with incremental updates.
    """
    def __init__(self):
        self.entries = []
        self.session_active = False
        self.session_name = ""
        self.last_summary = ""
        self.session_start_time = None
    
    def start_session(self, session_name):
        """Start a new summarization session"""
        self.session_active = True
        self.session_name = session_name
        self.entries = []
        self.last_summary = ""
        self.session_start_time = datetime.datetime.now()
    
    def end_session(self):
        """End the current session and save it"""
        if not self.session_active:
            return False
        
        # Save the session data
        session_data = {
            "name": self.session_name,
            "start_time": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entries": [{
                "text": entry["text"],
                "timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            } for entry in self.entries],
            "summary": self.last_summary
        }
        
        # Ensure sessions directory exists
        os.makedirs("saved_sessions", exist_ok=True)
        
        # Save to file
        filename = f"saved_sessions/{self.session_name}_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(session_data, f)
        
        # Reset session
        self.session_active = False
        self.entries = []
        self.session_name = ""
        self.last_summary = ""
        self.session_start_time = None
        
        return True
    
    def add_message(self, text):
        """Add a new message to the session"""
        if not self.session_active:
            return False
        
        self.entries.append({
            "text": text,
            "timestamp": datetime.datetime.now()
        })
        
        return True
    
    def get_entries(self):
        """Get all entries in the current session"""
        return self.entries
    
    def get_current_summary(self, summarizer=None):
        """Generate and return the current summary of all entries"""
        if not self.entries:
            return "No content to summarize yet."
        
        if summarizer:
            # Compile all entries
            all_text = "\n".join([entry["text"] for entry in self.entries])
            
            try:
                # Determine appropriate summary length based on content
                text_length = len(all_text)
                max_length = min(200, max(100, text_length // 4))  # Between 100-200 chars
                min_length = max(30, min(50, text_length // 10))   # Between 30-50 chars
                
                # Generate summary
                if text_length > 100:  # Only summarize if there's enough content
                    summary = summarizer(all_text, max_length=max_length, min_length=min_length, 
                                         do_sample=False)[0]['summary_text']
                    self.last_summary = summary
                    return summary
                else:
                    return "Session has insufficient content for meaningful summarization yet."
            except Exception as e:
                return f"Error generating summary: {str(e)}"
        else:
            # If no summarizer is provided, return the last generated summary or a message
            if self.last_summary:
                return self.last_summary
            else:
                return "Summarizer model not available."
    
    def extract_key_points(self, nlp=None, max_points=5):
        """Extract key points from all entries"""
        if not nlp or not self.entries:
            return []
        
        # Combine all text
        all_text = " ".join([entry["text"] for entry in self.entries])
        
        try:
            # Process with spaCy
            doc = nlp(all_text)
            
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Score sentences based on importance
            sentence_scores = {}
            for sent in sentences:
                # Score based on presence of named entities, length, and keywords
                if len(sent.split()) > 3:  # Ignore very short sentences
                    sent_doc = nlp(sent)
                    entity_count = len([ent for ent in sent_doc.ents])
                    keyword_count = len([token for token in sent_doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop])
                    
                    # Calculate score (adjust weights as needed)
                    score = (entity_count * 2) + (keyword_count * 1) - (0.1 * len(sent.split()))
                    sentence_scores[sent] = score
            
            # Get top sentences as key points
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_points]
            
            # Return just the sentences
            return [sent for sent, _ in top_sentences]
        except Exception as e:
            return []
    
    def is_session_active(self):
        """Check if there's an active session"""
        return self.session_active
    
    def get_session_name(self):
        """Get the name of the current session"""
        return self.session_name


class SummaryExplorer:
    """
    Manages saved summaries and provides tools for comparing and exploring them.
    """
    def __init__(self):
        self.sessions_dir = "saved_sessions"
        os.makedirs(self.sessions_dir, exist_ok=True)
    
    def get_saved_sessions(self):
        """Get list of all saved sessions"""
        if not os.path.exists(self.sessions_dir):
            return []
        
        # Get all JSON files in the sessions directory
        session_files = [f for f in os.listdir(self.sessions_dir) if f.endswith('.json')]
        
        # Extract session names
        session_names = []
        for filename in session_files:
            # Extract session name from filename (remove timestamp and extension)
            parts = filename.split('_')
            if len(parts) > 1:
                # Remove the timestamp part and .json extension
                name = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
                session_names.append(name)
        
        return session_names
    
    def load_session(self, session_name):
        """Load a specific session by name"""
        if not os.path.exists(self.sessions_dir):
            return None
        
        # Find files that match the session name
        session_files = [f for f in os.listdir(self.sessions_dir) 
                         if f.startswith(session_name) and f.endswith('.json')]
        
        if not session_files:
            return None
        
        # Load the most recent one (based on timestamp in filename)
        most_recent = sorted(session_files)[-1]
        
        try:
            with open(os.path.join(self.sessions_dir, most_recent), 'r') as f:
                session_data = json.load(f)
            
            return session_data
        except Exception as e:
            return None
    
    def compare_sessions(self, session_names):
        """Compare multiple sessions"""
        if not isinstance(session_names, list) or len(session_names) < 2:
            return None
        
        session_data = []
        for name in session_names:
            data = self.load_session(name)
            if data:
                session_data.append(data)
        
        if len(session_data) < 2:
            return None
        
        # Create comparison data
        comparison = {
            "sessions": [data["name"] for data in session_data],
            "summaries": [data["summary"] for data in session_data],
            "entry_counts": [len(data["entries"]) for data in session_data],
            "durations": []
        }
        
        # Calculate durations
        for data in session_data:
            try:
                start = datetime.datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M:%S")
                end = datetime.datetime.strptime(data["end_time"], "%Y-%m-%d %H:%M:%S")
                duration = (end - start).total_seconds() / 60  # Duration in minutes
                comparison["durations"].append(f"{duration:.1f} minutes")
            except:
                comparison["durations"].append("Unknown")
        
        return comparison


class PersonaSummarizer:
    """
    Generates summaries in different personas or styles.
    """
    def __init__(self):
        self.personas = {
            "teacher": "Explain concepts clearly and provide educational context",
            "executive": "Focus on key business insights and action items",
            "friend": "Use casual language and relate to everyday experiences",
            "journalist": "Present factual information in a structured, objective manner",
            "sarcastic": "Provide a humorous, slightly sarcastic take on the information",
            "steve_jobs": "Present information with passion and focus on innovation and design"
        }
    
    def get_available_personas(self):
        """Get list of available personas"""
        return list(self.personas.keys())
    
    def get_persona_description(self, persona_name):
        """Get description of a specific persona"""
        return self.personas.get(persona_name, "Standard summarization")
    
    def summarize_with_persona(self, text, persona, max_length=150, min_length=40, 
                               summarizer=None, zero_shot_model=None):
        """Generate a summary in the style of the selected persona"""
        if not summarizer:
            return "Summarization model not available."
        
        try:
            # First get a base summary
            base_summary = summarizer(text, max_length=max_length, min_length=min_length, 
                                     do_sample=False)[0]['summary_text']
            
            # Apply persona styling
            if persona == "teacher":
                return self._style_teacher(base_summary)
            elif persona == "executive":
                return self._style_executive(base_summary)
            elif persona == "friend":
                return self._style_friend(base_summary)
            elif persona == "journalist":
                return self._style_journalist(base_summary)
            elif persona == "sarcastic":
                return self._style_sarcastic(base_summary)
            elif persona == "steve_jobs":
                return self._style_steve_jobs(base_summary)
            else:
                return base_summary
                
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _style_teacher(self, summary):
        """Style summary as a teacher would explain it"""
        intro = "Let me explain this clearly: "
        
        # Add educational phrases
        educational_phrases = [
            "It's important to understand that ",
            "A key concept here is that ",
            "To put this in context, ",
            "This demonstrates how ",
        ]
        
        # Add an educational phrase at a suitable point
        sentences = summary.split('. ')
        if len(sentences) > 1:
            insert_point = min(1, len(sentences)-1)  # Insert after first sentence if possible
            phrase = np.random.choice(educational_phrases)
            sentences[insert_point] = phrase + sentences[insert_point].lower()
        
        # Add conclusion
        conclusion = " Remember, the main takeaway is that " + sentences[-1].lower()
        sentences[-1] = sentences[-1] + "."
        
        styled_summary = intro + '. '.join(sentences) + conclusion
        return styled_summary
    
    def _style_executive(self, summary):
        """Style summary for executives"""
        # Add business-oriented intro
        intro = "Executive Summary: "
        
        # Break into bullet points
        sentences = summary.split('. ')
        bullet_points = []
        
        # Create bullet points with business terms
        business_terms = ["Key finding", "Strategic point", "Market implication", 
                         "Actionable insight", "Bottom line"]
        
        for i, sentence in enumerate(sentences):
            if sentence:
                term = business_terms[i % len(business_terms)]
                bullet_points.append(f"• {term}: {sentence}")
        
        # Add action-oriented conclusion
        conclusion = "\nRecommended Action: Analyze these findings for strategic advantages."
        
        return intro + "\n\n" + "\n".join(bullet_points) + conclusion
    
    def _style_friend(self, summary):
        """Style summary as a friend would explain it"""
        # Add casual intro
        intros = [
            "So basically, ",
            "Hey, here's the deal: ",
            "Just so you know, ",
            "Check this out - "
        ]
        intro = np.random.choice(intros)
        
        # Make language more casual
        casual_summary = summary
        casual_summary = casual_summary.replace("therefore", "so")
        casual_summary = casual_summary.replace("subsequently", "then")
        casual_summary = casual_summary.replace("however", "but")
        casual_summary = casual_summary.replace("additionally", "also")
        
        # Add casual phrases
        casual_phrases = [" right?", ", you know?", " - pretty cool!", ", honestly."]
        sentences = casual_summary.split('. ')
        
        if len(sentences) > 2:
            insert_point = np.random.randint(0, len(sentences)-1)
            sentences[insert_point] = sentences[insert_point] + np.random.choice(casual_phrases)
        
        # Add friendly conclusion
        conclusions = [
            " Anyway, that's what's happening!",
            " That's the gist of it!",
            " Hope that makes sense!",
            " Crazy, right?"
        ]
        conclusion = np.random.choice(conclusions)
        
        styled_summary = intro + '. '.join(sentences) + conclusion
        return styled_summary
    
    def _style_journalist(self, summary):
        """Style summary as a journalist would write it"""
        # Add journalistic intro
        date = datetime.datetime.now().strftime("%B %d, %Y")
        intro = f"BREAKING NEWS ({date}) - "
        
        # Make language more objective and formal
        journalist_summary = summary
        
        # Structure like a news article with paragraphs
        sentences = journalist_summary.split('. ')
        paragraphs = []
        
        if len(sentences) >= 3:
            # First paragraph: first 1-2 sentences as lead
            lead_length = min(2, len(sentences) // 3)
            lead = '. '.join(sentences[:lead_length]) + '.'
            paragraphs.append(lead)
            
            # Second paragraph: next 2-3 sentences with details
            mid_length = min(3, len(sentences) // 2)
            mid = '. '.join(sentences[lead_length:lead_length+mid_length]) + '.'
            paragraphs.append(mid)
            
            # Final paragraph: remaining sentences
            remaining = '. '.join(sentences[lead_length+mid_length:])
            if remaining:
                paragraphs.append(remaining)
        else:
            paragraphs = [journalist_summary]
        
        # Add journalistic ending
        paragraphs[-1] += " We will continue to follow this story as it develops."
        
        styled_summary = intro + "\n\n" + "\n\n".join(paragraphs)
        return styled_summary
    
    def _style_sarcastic(self, summary):
        """Style summary with a sarcastic tone"""
        # Add sarcastic intro
        intros = [
            "Oh great, here's what you need to know: ",
            "Well, isn't this fascinating: ",
            "Prepare to be amazed by this groundbreaking information: ",
            "Hold onto your hats, folks: "
        ]
        intro = np.random.choice(intros)
        
        # Add sarcastic comments to sentences
        sarcastic_comments = [
            " (how original)",
            " (shocking, I know)",
            " (who would've guessed?)",
            " (revolutionary stuff here)",
            " (mind-blowing, right?)"
        ]
        
        sentences = summary.split('. ')
        for i in range(min(2, len(sentences))):
            insert_point = np.random.randint(0, len(sentences))
            sentences[insert_point] = sentences[insert_point] + np.random.choice(sarcastic_comments)
        
        # Add sarcastic conclusion
        conclusions = [
            " But what do I know, anyway?",
            " Clearly this changes everything.",
            " I'm sure we're all better for knowing this.",
            " File that under 'absolutely essential information'."
        ]
        conclusion = np.random.choice(conclusions)
        
        styled_summary = intro + '. '.join(sentences) + conclusion
        return styled_summary
    
    def _style_steve_jobs(self, summary):
        """Style summary as if Steve Jobs was presenting it"""
        # Add iconic intro
        intros = [
            "This is insanely great: ",
            "Today, I want to share something revolutionary: ",
            "This changes everything: ",
            "Here's something incredible: "
        ]
        intro = np.random.choice(intros)
        
        # Add emphasis on innovation and design
        steve_phrases = [
            " And it's amazing.",
            " It's that simple.",
            " It just works.",
            " And we think you're going to love it.",
            " This is the best implementation we've ever created."
        ]
        
        sentences = summary.split('. ')
        
        # Insert Steve-isms
        if len(sentences) > 1:
            for _ in range(min(2, len(sentences))):
                insert_point = np.random.randint(0, len(sentences))
                sentences[insert_point] = sentences[insert_point] + np.random.choice(steve_phrases)
        
        # Add his famous "one more thing" if appropriate
        if len(sentences) > 3:
            sentences.insert(-1, "And one more thing")
        
        styled_summary = intro + '. '.join(sentences)
        return styled_summary