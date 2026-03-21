# Program title: Sentiment-Aware Movie Review TTS
# Student names: [Your Name(s)]

import streamlit as st
from transformers import pipeline

# 在应用启动时预加载模型
@st.cache_resource
def load_sentiment_model():
    """预加载情感分析模型"""
    return pipeline("text-classification", model="megan21/roberta-finetune-movie-reviews-sentiment-analysis")

@st.cache_resource
def load_tts_model():
    """预加载TTS模型"""
    return pipeline("text-to-audio", model="facebook/mms-tts-eng")

# 初始化模型（应用启动时执行一次）
sentiment_pipeline = load_sentiment_model()
tts_pipeline = load_tts_model()

# function part
def analyze_sentiment(review):
    """Analyze sentiment of a movie review."""
    result = sentiment_pipeline(review)[0]
    return result['label'], result['score']

def text2audio(text):
    """Convert text to speech."""
    audio_data = tts_pipeline(text)
    return audio_data

def main():
    st.set_page_config(page_title="🎬 Sentiment-Aware Movie Review TTS", page_icon="🎬")
    st.header("🎬 Sentiment-Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud.")
    
    # 初始化会话状态
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'current_review' not in st.session_state:
        st.session_state.current_review = ""
    
    # User input
    user_review = st.text_area(
        "Your movie review:", 
        height=150, 
        placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.",
        key="review_input"
    )
    
    # 并排显示两个按钮
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
            if user_review.strip():
                st.session_state.analysis_done = True
                st.session_state.current_review = user_review
                st.rerun()  # 重新运行以显示结果
            else:
                st.warning("Please enter a movie review first.")
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.analysis_done = False
            st.session_state.current_review = ""
            st.rerun()
    
    # 分隔线
    st.divider()
    
    # 只有当有新的分析请求时才显示结果
    if st.session_state.analysis_done and user_review.strip():
        # 确保显示当前评论的结果
        if user_review == st.session_state.current_review:
            # Stage 1: Sentiment Analysis
            with st.spinner("Analyzing sentiment..."):
                label, score = analyze_sentiment(user_review)
            
            if label == "POSITIVE" or label == "LABEL_1":
                sentiment_display = f"**Sentiment:** 😊 Positive (Confidence: {score:.2%})"
            else:
                sentiment_display = f"**Sentiment:** 😞 Negative (Confidence: {score:.2%})"
            
            st.success("Analysis Complete!")
            st.markdown(sentiment_display)
            
            # Stage 2: Text to Audio
            with st.spinner("Generating audio..."):
                audio_data = text2audio(user_review)
            
            # 显示音频播放器
            st.subheader("🎧 Audio Preview")
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)
            
            # 下载音频按钮
            import numpy as np
            from scipy.io.wavfile import write
            import io
            
            # 将音频数据转换为WAV格式
            audio_int16 = np.int16(audio_array * 32767)
            wav_buffer = io.BytesIO()
            write(wav_buffer, sample_rate, audio_int16)
            
            st.download_button(
                label="⬇️ Download Audio",
                data=wav_buffer.getvalue(),
                file_name="movie_review_tts.wav",
                mime="audio/wav"
            )

if __name__ == "__main__":
    main()
