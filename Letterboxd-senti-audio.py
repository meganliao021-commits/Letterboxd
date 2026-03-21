# Program title: Sentiment-Aware Movie Review TTS
# Student names: [Your Name(s)]

import streamlit as st
from transformers import pipeline
import traceback


# function part
def analyze_sentiment(review):
    """Analyze sentiment of a movie review."""
    try:
        sentiment_pipeline = pipeline("text-classification", model="megan21/distilbert-base-uncased-finetuned-movie-review")
        result = sentiment_pipeline(review)[0]
        return result['label'], result['score'], None
    except Exception as e:
        return None, None, f"Failed to load sentiment model: {str(e)}"

def text2audio(text):
    """Convert text to speech."""
    try:
        tts_pipeline = pipeline("text-to-audio", model="facebook/mms-tts-eng")
        audio_data = tts_pipeline(text)
        return audio_data, None
    except Exception as e:
        return None, f"Failed to load TTS model: {str(e)}"

def main():
    st.set_page_config(page_title="🎬 Sentiment-Aware Movie Review TTS", page_icon="🎬")
    st.header("🎬 Sentiment-Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud.")

    # Initialize session state for storing results
    if 'result_available' not in st.session_state:
        st.session_state.result_available = False
    
    # User input
    user_review = st.text_area("Your movie review:", height=150, 
                                placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.",
                                key="review_input")

    # Create a container for results
    result_container = st.container()
    
    # Create analyze button
    analyze_clicked = st.button("🔍 Analyze", type="primary")
    
    if analyze_clicked and user_review:
        # Clear previous results in session state
        st.session_state.result_available = True
        
        with result_container:
            # Clear previous results by using empty containers
            sentiment_placeholder = st.empty()
            audio_placeholder = st.empty()
            
            # Stage 1: Sentiment Analysis
            sentiment_placeholder.text("Analyzing sentiment...")
            label, score, error = analyze_sentiment(user_review)
            
            if error:
                sentiment_placeholder.error(f"❌ Error: {error}")
            elif label:
                sentiment_placeholder.empty()  # Clear loading text
                if label == "POSITIVE" or label == "LABEL_1":
                    sentiment_placeholder.write(f"**Sentiment:** 😊 Positive (Confidence: {score:.2%})")
                else:
                    sentiment_placeholder.write(f"**Sentiment:** 😞 Negative (Confidence: {score:.2%})")
                
                # Stage 2: Text to Audio
                audio_placeholder.text("Generating audio...")
                audio_data, tts_error = text2audio(user_review)
                
                if tts_error:
                    audio_placeholder.error(f"❌ Error: {tts_error}")
                elif audio_data:
                    audio_placeholder.empty()  # Clear loading text
                    audio_array = audio_data["audio"]
                    sample_rate = audio_data["sampling_rate"]
                    audio_placeholder.write("**Audio Preview:**")
                    audio_placeholder.audio(audio_array, sample_rate=sample_rate)
                    
    elif analyze_clicked and not user_review:
        with result_container:
            st.warning("⚠️ Please enter a movie review first!")

if __name__ == "__main__":
    main()
