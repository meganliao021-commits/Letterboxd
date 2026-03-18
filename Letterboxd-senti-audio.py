
# --- 1. Imports ---
import streamlit as st
from transformers import pipeline
import torch
import soundfile as sf
import io
import numpy as np

# --- 2. Functions ---

@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    """Load the fine‑tuned RoBERTa sentiment model."""
    model_path = "megan21/roberta-finetune-movie-reviews-sentiment-analysis"
    # Adjust device if you have a GPU
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model_path, tokenizer=model_path, device=device)

@st.cache_resource(show_spinner="Loading TTS model...")
def load_tts_model():
    """Load a text‑to‑speech model that can handle emotional prompts."""
    # Using Bark-small for expressive control; you can replace with another model
    return pipeline("text-to-speech", model="suno/bark-small", device=0 if torch.cuda.is_available() else -1)

def analyze_sentiment(review_text, sentiment_pipe):
    """Return the label and confidence score for the review."""
    result = sentiment_pipe(review_text)[0]
    return result['label'], result['score']

def get_voice_prompt(sentiment_label, confidence=None):
    """
    Map sentiment to a descriptive voice prompt.
    Adjust the prompts below based on your model's output labels.
    """
    # Check for common label formats
    if sentiment_label in ["POSITIVE", "LABEL_1", "positive"]:
        return "A cheerful female speaker with an excited, expressive tone. She speaks at a moderately fast pace with clear audio."
    else:  # NEGATIVE, LABEL_0, negative
        return "A calm male speaker with a slightly disappointed tone. He speaks slowly and quietly."

def generate_speech(review_text, voice_prompt, tts_pipe):
    """Combine prompt and review, then generate audio."""
    full_text = voice_prompt + " " + review_text
    audio_output = tts_pipe(full_text, forward_params={"do_sample": True})
    return audio_output["audio"], audio_output["sampling_rate"]

# --- 3. Main App ---

def main():
    st.set_page_config(page_title="🎬 Movie Review Sentiment TTS", page_icon="🎬")
    st.title("🎬 Movie Review Sentiment & Speech Demo")
    st.markdown("Enter a movie review below. The app will detect its sentiment and read it aloud with an emotion‑matching voice.")

    # Load models (cached)
    sentiment_model = load_sentiment_model()
    tts_model = load_tts_model()

    # Text input
    user_review = st.text_area(
        "Your movie review:",
        height=150,
        placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.",
        help="Write or paste a movie review here."
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("🎤 Generate Speech", type="primary")

    if run_button and user_review.strip():
        # Step 1: Sentiment Analysis
        with st.status("🔍 Analyzing sentiment...", expanded=True) as status:
            label, score = analyze_sentiment(user_review, sentiment_model)
            status.update(label="Sentiment analysis complete!", state="complete")
        
        # Display result
        if label in ["POSITIVE", "LABEL_1"]:
            st.metric("Sentiment", "😊 Positive", f"{score:.2%} confidence")
        else:
            st.metric("Sentiment", "😞 Negative", f"{score:.2%} confidence")
        
        # Step 2: Prepare voice prompt
        voice_prompt = get_voice_prompt(label, score)
        st.caption(f"🎤 Voice style: {voice_prompt.strip()}")

        # Step 3: Generate speech
        with st.status("🎧 Generating speech...", expanded=True) as status:
            audio_array, sampling_rate = generate_speech(user_review, voice_prompt, tts_model)
            status.update(label="Speech ready!", state="complete")

        # Step 4: Play audio
        st.subheader("🔊 Listen to the review")
        # Convert numpy array to bytes for Streamlit audio
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, samplerate=sampling_rate, format="wav")
        buffer.seek(0)
        st.audio(buffer, format="audio/wav")

        # Optional download button
        st.download_button(
            label="📥 Download as WAV",
            data=buffer,
            file_name="review_speech.wav",
            mime="audio/wav"
        )

    elif run_button and not user_review.strip():
        st.warning("Please enter a review first.")

    # Example reviews (optional)
    with st.expander("💡 Try an example"):
        examples = [
            "An absolute masterpiece! The direction, acting, and score all come together perfectly.",
            "Terrible movie. Waste of time. The plot made no sense and the acting was wooden.",
            "It was okay. Some good moments but overall forgettable."
        ]
        for i, ex in enumerate(examples):
            if st.button(f"Load Example {i+1}", key=f"ex_{i}"):
                st.session_state["review_text"] = ex
                st.rerun()

if __name__ == "__main__":
    main()
