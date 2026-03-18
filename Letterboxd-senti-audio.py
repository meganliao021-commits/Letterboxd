# Program: Sentiment‑Aware Movie Review TTS
# Student names: [Your Name(s)]

import streamlit as st
from transformers import pipeline
import soundfile as sf
import io
import torch

# ---------- Load models with caching ----------
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    """Load your fine‑tuned RoBERTa model for movie review sentiment."""
    return pipeline(
        "text-classification",
        model="megan21/roberta-finetune-movie-reviews-sentiment-analysis",
        device=0 if torch.cuda.is_available() else -1
    )

@st.cache_resource(show_spinner="Loading TTS model...")
def load_tts_model():
    """Load a TTS model that supports voice style control (Parler‑TTS)."""
    return pipeline(
        "text-to-speech",
        model="parler-tts/parler_tts_mini_v0.1",
        device=0 if torch.cuda.is_available() else -1
    )

# ---------- Helper: map sentiment to voice preset ----------
def get_voice_preset(sentiment_label):
    """Return a natural language voice description matching the sentiment."""
    if sentiment_label in ["POSITIVE", "LABEL_1", "positive"]:
        return "A cheerful female speaker with an excited, expressive tone. She speaks at a moderately fast pace with clear audio."
    else:  # NEGATIVE, LABEL_0, negative
        return "A calm male speaker with a slightly disappointed tone. He speaks slowly and quietly."

# ---------- Main app ----------
def main():
    st.set_page_config(page_title="🎬 Sentiment‑Aware TTS", page_icon="🎬")
    st.title("🎬 Sentiment‑Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud with a matching voice.")

    # Load models
    sentiment_model = load_sentiment_model()
    tts_model = load_tts_model()

    # Text input area
    user_review = st.text_area(
        "Your movie review:",
        height=150,
        placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking and the acting superb."
    )

    # Generate button – only runs when clicked
    if st.button("🎤 Generate speech", type="primary") and user_review.strip():
        # 1. Sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            result = sentiment_model(user_review)[0]
            label = result["label"]
            score = result["score"]

        # Display result
        if label in ["POSITIVE", "LABEL_1"]:
            st.metric("Sentiment", "😊 Positive", f"{score:.2%} confidence")
        else:
            st.metric("Sentiment", "😞 Negative", f"{score:.2%} confidence")

        # 2. Choose voice preset based on sentiment
        voice_preset = get_voice_preset(label)
        st.caption(f"🎤 Voice style: {voice_preset}")

        # 3. Generate speech using Parler‑TTS (pass voice_preset as a parameter)
        with st.spinner("Generating speech... (may take a few seconds)"):
            audio_output = tts_model(user_review, voice_preset=voice_preset)
            audio_array = audio_output["audio"]
            sampling_rate = audio_output["sampling_rate"]

        # 4. Play audio
        st.subheader("🔊 Listen to the review")
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

    elif st.button("🎤 Generate speech", type="primary") and not user_review.strip():
        st.warning("Please enter a review first.")

    # Example reviews for quick testing
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
