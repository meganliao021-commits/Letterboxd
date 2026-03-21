# Program title: Sentiment-Aware Movie Review TTS
# Student names: [Your Name(s)]


import streamlit as st
from transformers import pipeline


# function part
def analyze_sentiment(review):
    """Analyze sentiment of a movie review."""
    sentiment_pipeline = pipeline("text-classification", model="megan21/distilbert-base-uncased-finetuned-movie-review")
    result = sentiment_pipeline(review)[0]
    return result['label'], result['score']

def text2audio(text):
    """Convert text to speech."""
    tts_pipeline = pipeline("text-to-audio", model="facebook/mms-tts-eng")
    audio_data = tts_pipeline(text)
    return audio_data

def main():
    st.set_page_config(page_title="🎬 Sentiment-Aware Movie Review TTS", page_icon="🎬")
    st.header("🎬 Sentiment-Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud.")

    # User input
    user_review = st.text_area("Your movie review:", height=150, 
                                placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.")

    if user_review:
        # Stage 1: Sentiment Analysis
        st.text("Analyzing sentiment...")
        label, score = analyze_sentiment(user_review)
        if label == "POSITIVE" or label == "LABEL_1":
            st.write(f"**Sentiment:** 😊 Positive (Confidence: {score:.2%})")
        else:
            st.write(f"**Sentiment:** 😞 Negative (Confidence: {score:.2%})")

        # Stage 2: Text to Audio
        st.text("Generating audio...")
        audio_data = text2audio(user_review)

        # Play button
        if st.button("🔊 Play Audio"):
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)

if __name__ == "__main__":
    main()
