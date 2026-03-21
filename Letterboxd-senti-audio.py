# Program title: Sentiment-Aware Movie Review TTS
# Student names: [Your Name(s)]

import streamlit as st
from transformers import pipeline
import sys

# 在应用启动时全局加载模型
@st.cache_resource(show_spinner=False)
def load_models():
    """一次性加载所有需要的模型"""
    try:
        with st.spinner("Loading sentiment analysis model..."):
            sentiment_pipeline = pipeline("text-classification", 
                                         model="megan21/roberta-finetune-movie-reviews-sentiment-analysis")
        
        with st.spinner("Loading text-to-speech model..."):
            tts_pipeline = pipeline("text-to-audio", 
                                   model="facebook/mms-tts-eng")
        
        st.success("✅ Models loaded successfully!")
        return sentiment_pipeline, tts_pipeline
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.stop()
        return None, None

# 全局变量存储加载的模型
if 'models_loaded' not in st.session_state:
    sentiment_pipeline, tts_pipeline = load_models()
    if sentiment_pipeline is not None and tts_pipeline is not None:
        st.session_state.models_loaded = True
        st.session_state.sentiment_pipeline = sentiment_pipeline
        st.session_state.tts_pipeline = tts_pipeline
    else:
        st.session_state.models_loaded = False

# function part
def analyze_sentiment(review):
    """Analyze sentiment of a movie review."""
    if not st.session_state.models_loaded:
        st.error("Models are not loaded properly. Please refresh the page.")
        return None, None
    
    result = st.session_state.sentiment_pipeline(review)[0]
    return result['label'], result['score']

def text2audio(text):
    """Convert text to speech."""
    if not st.session_state.models_loaded:
        st.error("Models are not loaded properly. Please refresh the page.")
        return None
    
    audio_data = st.session_state.tts_pipeline(text)
    return audio_data

def clear_previous_output():
    """Clear previous sentiment and audio output"""
    if 'last_review' in st.session_state:
        # 清除之前的结果
        st.session_state.pop('sentiment_result', None)
        st.session_state.pop('audio_data', None)
        st.session_state.pop('audio_generated', None)

def main():
    st.set_page_config(page_title="🎬 Sentiment-Aware Movie Review TTS", page_icon="🎬")
    st.header("🎬 Sentiment-Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud.")
    
    # 初始化session state
    if 'audio_generated' not in st.session_state:
        st.session_state.audio_generated = False
    if 'last_review' not in st.session_state:
        st.session_state.last_review = ""
    if 'processing' not in st.session_state:
        st.session_state.processing = False

    # User input with on_change handler
    user_review = st.text_area(
        "Your movie review:", 
        height=150, 
        placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.",
        key="review_input",
        on_change=clear_previous_output
    )
    
    # 检查是否按下Ctrl+Enter
    if user_review and user_review != st.session_state.last_review:
        # 清除之前的结果
        clear_previous_output()
        st.session_state.last_review = user_review
        st.session_state.processing = True
        
        # 添加一个重新分析按钮
        if st.button("🔄 Analyze & Generate Audio"):
            st.session_state.processing = True
        else:
            st.session_state.processing = False
    
    # 只有当用户点击按钮或按Ctrl+Enter时才处理
    if user_review and st.session_state.processing:
        # Stage 1: Sentiment Analysis with progress
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing sentiment..."):
            label, score = analyze_sentiment(user_review)
        
        if label and score:
            # 保存结果到session state
            st.session_state.sentiment_result = (label, score)
            
            # Display sentiment
            if label == "POSITIVE" or label == "LABEL_1":
                st.write(f"**Sentiment:** 😊 Positive (Confidence: {score:.2%})")
            else:
                st.write(f"**Sentiment:** 😞 Negative (Confidence: {score:.2%})")

            # Stage 2: Text to Audio with progress bar
            st.text("Generating audio...")
            progress_bar = st.progress(0)
            
            with st.spinner("Synthesizing speech..."):
                audio_data = text2audio(user_review)
                progress_bar.progress(100)
            
            if audio_data:
                # 保存音频数据到session state
                st.session_state.audio_data = audio_data
                st.session_state.audio_generated = True
                
                # 播放按钮
                if st.button("🔊 Play Audio"):
                    audio_array = audio_data["audio"]
                    sample_rate = audio_data["sampling_rate"]
                    st.audio(audio_array, sample_rate=sample_rate)
                    
                    # 添加下载按钮
                    st.download_button(
                        label="💾 Download Audio",
                        data=audio_array.tobytes(),
                        file_name="movie_review_audio.wav",
                        mime="audio/wav"
                    )
    
    # 如果用户清除了输入
    elif not user_review and st.session_state.last_review:
        st.session_state.last_review = ""
        st.session_state.audio_generated = False
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    # 检查模型是否加载成功
    if 'models_loaded' in st.session_state and not st.session_state.models_loaded:
        st.error("""
        ⚠️ Application failed to start properly.
        
        Possible reasons:
        1. Internet connection is required to download models
        2. Models may be temporarily unavailable
        3. You may need to install additional dependencies
        
        Please refresh the page to try again.
        """)
    else:
        main()
