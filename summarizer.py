import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Real-time summarization components
class RealTimeSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """Initialize the real-time summarizer with a specified model."""
        self.message_buffer = []
        self.last_summary_time = datetime.now()
        self.summary_history = []
        
        # Load model with error handling
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error loading summarization model: {e}")
            self.model_loaded = False
    
    def add_message(self, message_data):
        """Add a new message to the buffer and potentially trigger summarization."""
        if not isinstance(message_data, dict):
            message_data = {"speaker": "Unknown", "message": str(message_data), "timestamp": datetime.now()}
        elif "timestamp" not in message_data:
            message_data["timestamp"] = datetime.now()
            
        self.message_buffer.append(message_data)
        
        # Check if we should generate a new summary
        time_since_last_summary = datetime.now() - self.last_summary_time
        if time_since_last_summary.total_seconds() > 60 or len(self.message_buffer) >= 20:
            self.generate_incremental_summary()
            
    def generate_incremental_summary(self):
        """Generate a summary of recent messages and update summary history."""
        if not self.message_buffer or not self.model_loaded:
            return None
            
        # Combine recent messages for summarization
        recent_text = " ".join([f"{msg['speaker']}: {msg['message']}" for msg in self.message_buffer])
        
        # Generate summary
        try:
            summary = self.summarizer(recent_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            
            # Store summary with timestamp
            summary_entry = {
                "timestamp": datetime.now(),
                "summary": summary,
                "message_count": len(self.message_buffer),
                "participants": list(set([msg['speaker'] for msg in self.message_buffer])),
                "original_messages": self.message_buffer.copy()
            }
            
            self.summary_history.append(summary_entry)
            self.last_summary_time = datetime.now()
            self.message_buffer = []  # Clear buffer after summarization
            
            return summary_entry
        except Exception as e:
            st.error(f"Error generating incremental summary: {e}")
            return None
    
    def get_unseen_highlights(self, last_seen_timestamp):
        """Extract important highlights from messages that were not seen by the user."""
        if not self.model_loaded:
            return "Summary model not available."
            
        # Find messages since the last time the user checked
        unseen_messages = [msg for msg in self.message_buffer 
                         if msg['timestamp'] > last_seen_timestamp]
        
        if not unseen_messages:
            return "No new messages since you last checked."
        
        # For important messages, we'll use a basic heuristic (this could be enhanced)
        # Check for questions, action items, decisions, etc.
        important_messages = []
        for msg in unseen_messages:
            text = msg['message'].lower()
            
            # Simple importance detection based on keywords and patterns
            if (any(kw in text for kw in ["urgent", "important", "critical", "asap", "deadline"]) or
                text.endswith("?") or
                any(kw in text for kw in ["please", "need to", "must", "should we", "action item"])):
                important_messages.append(msg)
        
        # Format highlights
        if important_messages:
            highlights = "**Important Unseen Messages:**\n\n"
            for msg in important_messages:
                time_str = msg['timestamp'].strftime("%H:%M")
                highlights += f"• {time_str} - **{msg['speaker']}:** {msg['message']}\n"
            return highlights
        else:
            # Just return the number of new messages
            return f"*{len(unseen_messages)} new messages since you last checked.*"

# Conversational summary backtracking
class SummaryExplorer:
    def __init__(self, summary_history=None):
        """Initialize the summary explorer with existing summary history if available."""
        self.summary_history = summary_history if summary_history else []
        
        # Load a model for answering questions about summaries
        try:
            self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            self.model_loaded = True
        except Exception as e:
            st.error(f"Error loading question-answering model: {e}")
            self.model_loaded = False
    
    def answer_question(self, question, context=None):
        """Answer questions about the summary history."""
        if not self.model_loaded:
            return "I can't answer questions about the summary right now."
            
        # If no specific context is provided, use the entire summary history
        if not context:
            if not self.summary_history:
                return "There's no conversation history to answer questions about yet."
            
            # Combine recent summaries to create context
            context = " ".join([entry["summary"] for entry in self.summary_history[-5:]])
        
        # Process specific question types with special handling
        question_lower = question.lower()
        
        # Questions about disagreements
        if "who disagreed" in question_lower or "disagree with" in question_lower:
            return self._handle_disagreement_question(question)
        
        # Questions about decisions
        elif "what led to" in question_lower or "why was" in question_lower:
            return self._handle_decision_question(question)
        
        # Questions about participants
        elif "who said" in question_lower or "who mentioned" in question_lower:
            return self._handle_participant_question(question)
        
        # General questions - use the QA model
        try:
            answer = self.qa_model(question=question, context=context)
            
            # Check confidence score
            if answer['score'] < 0.1:
                return "I'm not confident about the answer based on available context."
            
            return answer['answer']
        except Exception as e:
            return f"I couldn't answer that question: {str(e)}"

    def _handle_disagreement_question(self, question):
        """Special handling for questions about disagreements."""
        # Look through original messages for disagreement patterns
        disagreements = []
        
        for entry in self.summary_history:
            messages = entry["original_messages"]
            
            # Look for patterns of disagreement between speakers
            for i, msg1 in enumerate(messages[:-1]):
                for msg2 in messages[i+1:]:
                    if msg1['speaker'] != msg2['speaker']:
                        text1 = msg1['message'].lower()
                        text2 = msg2['message'].lower()
                        
                        # Check for disagreement patterns
                        disagreement_indicators = [
                            "disagree", "not true", "incorrect", "i don't think", 
                            "that's wrong", "no,", "actually,", "but ", "however,"
                        ]
                        
                        if any(ind in text2 for ind in disagreement_indicators):
                            # Check if the second message references the first speaker
                            if any(term in text2 for term in [msg1['speaker'].lower(), "you"]):
                                disagreements.append((msg1, msg2))
        
        if disagreements:
            response = "I found these disagreements:\n\n"
            for msg1, msg2 in disagreements[:3]:  # Limit to 3 disagreements
                response += f"**{msg1['speaker']}:** {msg1['message']}\n"
                response += f"**{msg2['speaker']}:** {msg2['message']}\n\n"
            return response
        else:
            return "I couldn't find any clear disagreements in the conversation history."

    def _handle_decision_question(self, question):
        """Special handling for questions about decisions or reasoning."""
        # Look for messages that might indicate decisions
        decision_indicators = ["decided", "concluded", "agreed", "will", "plan", "going to"]
        
        relevant_messages = []
        for entry in self.summary_history:
            for msg in entry["original_messages"]:
                msg_lower = msg['message'].lower()
                if any(ind in msg_lower for ind in decision_indicators):
                    relevant_messages.append(msg)
        
        if relevant_messages:
            response = "These messages may explain the decision process:\n\n"
            for msg in relevant_messages[:5]:  # Limit to 5 messages
                response += f"**{msg['speaker']}:** {msg['message']}\n\n"
            return response
        else:
            return "I couldn't find clear information about how this decision was made."

    def _handle_participant_question(self, question):
        """Special handling for questions about specific participants."""
        # Extract the topic they're asking about
        topic_match = re.search(r"who (said|mentioned) (about |)(.+?)(\?|$)", question.lower())
        if not topic_match:
            return "I'm not sure what topic you're asking about."
        
        topic = topic_match.group(3).strip()
        
        # Find messages mentioning this topic
        relevant_messages = []
        for entry in self.summary_history:
            for msg in entry["original_messages"]:
                if topic in msg['message'].lower():
                    relevant_messages.append(msg)
        
        if relevant_messages:
            response = f"These people mentioned '{topic}':\n\n"
            for msg in relevant_messages[:3]:  # Limit to 3 messages
                response += f"**{msg['speaker']}:** {msg['message']}\n\n"
            return response
        else:
            return f"I couldn't find anyone who specifically mentioned '{topic}'."

# Persona-based summarization styles
class PersonaSummarizer:
    def __init__(self):
        """Initialize the persona-based summarizer with different styles."""
        self.personas = {
            "teacher": {
                "description": "Clear, educational summary with key lessons",
                "prefix": "Today's key learnings: ",
                "style_guide": "Break down complex ideas into simple parts. Use clear, educational language. Emphasize key takeaways and lessons.",
                "example": "Our meeting covered three essential topics: (1) the quarterly budget, where we identified a 15% increase in marketing spend, (2) the product roadmap, which prioritizes mobile features, and (3) team restructuring, which will enhance cross-functional collaboration."
            },
            "executive": {
                "description": "Brief, business-focused summary highlighting decisions and next steps",
                "prefix": "Executive Summary: ",
                "style_guide": "Be concise and direct. Focus on decisions, action items, and business impact. Use precise language and avoid unnecessary details.",
                "example": "Decision made to proceed with Project Alpha (budget $1.2M). Marketing campaign launching next quarter. Team expansion planned for Q3. Projected ROI: 25% within 6 months."
            },
            "friend": {
                "description": "Casual, conversational summary using informal language",
                "prefix": "So basically, ",
                "style_guide": "Use casual, conversational language. Include conversational markers (you know, like, etc). Focus on the interesting parts and personality dynamics.",
                "example": "So basically, the team spent half the meeting arguing about the logo colors (classic marketing vs. design drama!). Then Alex swooped in with that market research we needed, and everyone finally agreed on the blue option. Oh, and heads up - we're all doing that team building thing next Friday, so clear your calendar!"
            },
            "journalist": {
                "description": "Factual, balanced summary with key quotes",
                "prefix": "Report: ",
                "style_guide": "Present balanced facts. Include 'quotes' from key participants. Organize by importance (inverted pyramid style). Be objective and comprehensive.",
                "example": "The city council meeting addressed three main issues: First, the controversial downtown development was approved by a 4-3 vote. Mayor Johnson called it 'a crucial step for economic growth,' while Councilor Smith expressed concerns about 'inadequate community input.' Second, the budget amendment for park renovations passed unanimously. Finally, the council postponed the vote on parking regulations until next month."
            },
            "sarcastic": {
                "description": "Humorous, sarcastic summary with witty observations",
                "prefix": "Oh great, another meeting where ",
                "style_guide": "Use sarcasm and irony. Point out contradictions and absurdities. Include witty observations. Exaggerate for humorous effect.",
                "example": "Oh great, another meeting where everyone pretended to understand the technical jargon Tom was throwing around. Highlights included Sarah volunteering everyone else for weekend work, the coffee running out precisely when the budget discussion started (coincidence? I think not), and the classic 'let's circle back' on every decision that actually mattered. Looking forward to doing it all again next week!"
            },
            "steve_jobs": {
                "description": "Visionary, product-focused summary with bold statements",
                "prefix": "Here's what's insanely great: ",
                "style_guide": "Use simple, powerful language. Make bold, decisive statements. Focus on product excellence and user experience. Emphasize 'revolutionary' aspects.",
                "example": "Here's what's insanely great: We've created something extraordinary. Our new approach completely reimagines how teams communicate. It's not just incremental improvement - it's a revolution in collaboration. The interface is so intuitive that training becomes unnecessary. This will fundamentally change how people work together. It's that simple. It's that powerful."
            }
        }
    
    def get_available_personas(self):
        """Return the list of available personas."""
        return list(self.personas.keys())
    
    def get_persona_description(self, persona_key):
        """Return the description for a specific persona."""
        if persona_key in self.personas:
            return self.personas[persona_key]["description"]
        return "Unknown persona style"
    
    def summarize_with_persona(self, text, persona_key, max_length=200):
        """Generate a summary using a specific persona style."""
        if persona_key not in self.personas:
            return f"Error: Unknown persona style '{persona_key}'"
            
        try:
            # First generate a base summary
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            base_summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
            
            # Now transform it to match the persona style
            persona = self.personas[persona_key]
            
            # Apply persona-specific transformations to the base summary
            if persona_key == "teacher":
                styled_summary = self._apply_teacher_style(base_summary)
            elif persona_key == "executive":
                styled_summary = self._apply_executive_style(base_summary)
            elif persona_key == "friend":
                styled_summary = self._apply_friend_style(base_summary)
            elif persona_key == "journalist":
                styled_summary = self._apply_journalist_style(base_summary)
            elif persona_key == "sarcastic":
                styled_summary = self._apply_sarcastic_style(base_summary)
            elif persona_key == "steve_jobs":
                styled_summary = self._apply_steve_jobs_style(base_summary)
            else:
                styled_summary = base_summary
            
            # Add the persona prefix
            final_summary = persona["prefix"] + styled_summary
            
            return final_summary
        except Exception as e:
            return f"Error generating summary with persona '{persona_key}': {str(e)}"
    
    def _apply_teacher_style(self, text):
        """Transform text to match a teacher's explanatory style."""
        # Add numbering to make it more structured
        sentences = text.split('. ')
        if len(sentences) > 3:
            # Add numerical markers
            structured_text = ""
            for i, sentence in enumerate(sentences[:3], 1):
                structured_text += f"({i}) {sentence}. "
            if len(sentences) > 3:
                structured_text += " ".join(sentences[3:])
            return structured_text
        
        # Add educational phrases
        educational_markers = ["importantly", "note that", "remember", "key point"]
        if len(text) > 100:
            marker = np.random.choice(educational_markers)
            insert_point = len(text) // 2
            text = text[:insert_point] + f" {marker.capitalize()}, " + text[insert_point:]
        
        return text
    
    def _apply_executive_style(self, text):
        """Transform text to match an executive's direct style."""
        # Make sentences shorter and more direct
        sentences = re.split(r'(?<=[.!?])\s+', text)
        shorter_sentences = []
        
        for sentence in sentences:
            # Simplify long sentences
            if len(sentence.split()) > 15:
                words = sentence.split()
                sentence = " ".join(words[:12]) + "."
            shorter_sentences.append(sentence)
        
        # Add business-focused framing
        business_terms = ["ROI", "strategy", "results", "objectives", "timeline", "milestones"]
        enhanced_text = " ".join(shorter_sentences)
        
        # Insert business term if text is substantial
        if len(enhanced_text) > 100:
            term = np.random.choice(business_terms)
            insert_point = len(enhanced_text) // 2
            enhanced_text = enhanced_text[:insert_point] + f" {term}: " + enhanced_text[insert_point:]
        
        return enhanced_text
    
    def _apply_friend_style(self, text):
        """Transform text to match a casual, friendly style."""
        # Add casual markers
        casual_markers = ["like", "you know", "basically", "kinda", "pretty much", "honestly"]
        fillers = ["um", "uh", "so yeah", "I mean"]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        casual_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 3:  # Skip very short sentences
                casual_sentences.append(sentence)
                continue
                
            if i == 0 or np.random.random() > 0.7:
                # Add casual marker to some sentences
                marker = np.random.choice(casual_markers)
                sentence = f"{sentence} {marker}"
            
            if np.random.random() > 0.8:
                # Add filler to some sentences
                filler = np.random.choice(fillers)
                sentence = f"{filler}, {sentence}"
                
            casual_sentences.append(sentence)
        
        # Add informal punctuation
        result = " ".join(casual_sentences)
        if np.random.random() > 0.5:
            result = result.replace(".", "...").replace("?", "??")
        
        return result
    
    def _apply_journalist_style(self, text):
        """Transform text to match a journalistic style."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        journalistic_text = []
        
        # Identify potential quotes
        for i, sentence in enumerate(sentences):
            # Add attribution to some sentences
            if i > 0 and len(sentence) > 10 and np.random.random() > 0.7:
                names = ["Smith", "Johnson", "Chen", "Rodriguez", "Patel", "Kim"]
                titles = ["Director", "Manager", "Analyst", "Spokesperson", "Representative"]
                name = np.random.choice(names)
                title = np.random.choice(titles)
                
                # Turn into a quote
                sentence = f'"{sentence}" said {name}, {title}.'
            
            journalistic_text.append(sentence)
        
        result = " ".join(journalistic_text)
        
        # Add a journalistic lead if text is substantial
        if len(result) > 100:
            lead_phrases = [
                "According to sources, ",
                "Recent developments indicate that ",
                "In a surprising turn of events, ",
                "Analysis shows that ",
            ]
            lead = np.random.choice(lead_phrases)
            result = lead + result
        
        return result
    
    def _apply_sarcastic_style(self, text):
        """Transform text to match a sarcastic style."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sarcastic_sentences = []
        
        sarcastic_phrases = [
            "surprise, surprise, ",
            "shock and awe, ",
            "as if we didn't see that coming, ",
            "in a completely predictable turn of events, ",
            "brace yourselves for this shocking revelation: ",
            "hold on to your hats, "
        ]
        
        exaggerations = [
            "absolutely brilliant",
            "pure genius",
            "revolutionary",
            "mind-blowing",
            "earth-shattering"
        ]
        
        for i, sentence in enumerate(sentences):
            if i == 0 or np.random.random() > 0.7:
                # Add sarcastic opener to some sentences
                phrase = np.random.choice(sarcastic_phrases)
                sentence = phrase + sentence[0].lower() + sentence[1:]
            
            # Replace positive adjectives with exaggerated versions
            for adj in ["good", "nice", "effective", "useful"]:
                if adj in sentence.lower():
                    exaggeration = np.random.choice(exaggerations)
                    sentence = re.sub(r'\b' + adj + r'\b', exaggeration, sentence, flags=re.IGNORECASE)
            
            sarcastic_sentences.append(sentence)
        
        result = " ".join(sarcastic_sentences)
        
        # Add sarcastic conclusion
        if np.random.random() > 0.5:
            conclusions = [
                " Can't wait to see how this one plays out.",
                " What could possibly go wrong?",
                " I'm sure everyone's thrilled about that.",
                " Another day, another brilliant plan."
            ]
            result += np.random.choice(conclusions)
        
        return result
    
    def _apply_steve_jobs_style(self, text):
        """Transform text to match Steve Jobs' visionary style."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        jobs_sentences = []
        
        jobs_phrases = [
            "It's amazing. ",
            "This changes everything. ",
            "It's incredible. ",
            "Think about it. ",
            "It's that simple. ",
            "It's that revolutionary. "
        ]
        
        superlatives = [
            "extraordinary",
            "revolutionary",
            "magical",
            "incredible",
            "insanely great"
        ]
        
        for i, sentence in enumerate(sentences):
            # Replace descriptive adjectives with superlatives
            for adj in ["good", "great", "new", "improved", "better"]:
                if adj in sentence.lower():
                    superlative = np.random.choice(superlatives)
                    sentence = re.sub(r'\b' + adj + r'\b', superlative, sentence, flags=re.IGNORECASE)
            
            if i == len(sentences) - 1 or np.random.random() > 0.7:
                # Add Jobs-style phrase to some sentences
                phrase = np.random.choice(jobs_phrases)
                jobs_sentences.append(sentence)
                jobs_sentences.append(phrase)
            else:
                jobs_sentences.append(sentence)
        
        result = " ".join(jobs_sentences)
        
        # Add a Jobs-style comparison
        if len(result) > 100:
            comparisons = [
                " It's like nothing you've ever experienced before.",
                " It makes everything else look primitive by comparison.",
                " It's not just a little better. It's in a completely different league.",
                " We've reinvented the way this works."
            ]
            result += np.random.choice(comparisons)
        
        return result