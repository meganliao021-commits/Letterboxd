# Program title: Sentiment-Aware Movie Review TTS
# Student names: [Your Name(s)]

import streamlit as st
from transformers import pipeline
import time

# 全局变量存储加载的模型
sentiment_pipeline = None
tts_pipeline = None
model_loaded = False
model_error = None

def load_models():
    """在应用启动时一次性加载所有模型"""
    global sentiment_pipeline, tts_pipeline, model_loaded, model_error
    
    model_error = None
    try:
        # 使用进度条显示加载过程
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 加载情感分析模型
        status_text.text("Loading sentiment analysis model...")
        sentiment_pipeline = pipeline("text-classification", model="megan21/roberta-finetune-movie-reviews-sentiment-analysis")
        progress_bar.progress(50)
        
        # 加载TTS模型
        status_text.text("Loading text-to-speech model...")
        tts_pipeline = pipeline("text-to-audio", model="facebook/mms-tts-eng")
        progress_bar.progress(100)
        
        status_text.text("Models loaded successfully!")
        time.sleep(0.5)  # 短暂显示成功消息
        status_text.empty()
        progress_bar.empty()
        
        model_loaded = True
        return True
        
    except Exception as e:
        model_error = str(e)
        model_loaded = False
        return False

# 修改原有函数，使用预加载的模型
def analyze_sentiment(review):
    """Analyze sentiment of a movie review using pre-loaded model."""
    if not model_loaded:
        raise Exception("Model not loaded. Please restart the application.")
    
    result = sentiment_pipeline(review)[0]
    return result['label'], result['score']

def text2audio(text):
    """Convert text to speech using pre-loaded model."""
    if not model_loaded:
        raise Exception("Model not loaded. Please restart the application.")
    
    audio_data = tts_pipeline(text)
    return audio_data

def main():
    st.set_page_config(page_title="🎬 Sentiment-Aware Movie Review TTS", page_icon="🎬")
    st.header("🎬 Sentiment-Aware Movie Review TTS")
    st.write("Enter a movie review. The app will detect its sentiment and read it aloud.")
    
    # 初始化session state用于存储当前评论
    if 'current_review' not in st.session_state:
        st.session_state.current_review = ""
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # 检查模型是否已加载，如果没有则加载
    global model_loaded, model_error
    if not model_loaded:
        with st.spinner("Initializing application..."):
            load_models()
    
    # 显示模型加载错误（如果有）
    if model_error:
        st.error(f"⚠️ Model loading failed: {model_error}")
        st.info("Please check your internet connection and try refreshing the page.")
        return
    
    # User input with on_change callback
    user_review = st.text_area(
        "Your movie review:", 
        height=150, 
        placeholder="e.g., This film was absolutely amazing! The cinematography was breathtaking.",
        key="review_input",
        on_change=lambda: update_review(st.session_state.review_input)
    )
    
    # 当有新的评论输入时，清除之前的显示结果
    def update_review(new_review):
        if new_review != st.session_state.current_review:
            st.session_state.current_review = new_review
            st.session_state.show_results = False
    
    # 只有当用户按下Ctrl+Enter或主动提交时才显示结果
    col1, col2 = st.columns([6, 1])
    with col1:
        st.caption("Press Ctrl+Enter to submit, or use the button below")
    with col2:
        if st.button("Submit", type="primary"):
            if user_review and user_review.strip():
                st.session_state.show_results = True
    
    # 显示提交提示
    if user_review and not st.session_state.show_results:
        st.info("📝 Review entered. Press Ctrl+Enter or click Submit to analyze.")
    
    # 只有当show_results为True时才显示分析结果
    if user_review and st.session_state.show_results and model_loaded:
        # Stage 1: Sentiment Analysis with progress
        st.subheader("📊 Analysis Results")
        
        with st.spinner("Analyzing sentiment..."):
            label, score = analyze_sentiment(user_review)
        
        if label == "POSITIVE" or label == "LABEL_1":
            st.success(f"**Sentiment:** 😊 Positive (Confidence: {score:.2%})")
        else:
            st.error(f"**Sentiment:** 😞 Negative (Confidence: {score:.2%})")
        
        # Stage 2: Text to Audio with progress
        st.subheader("🔊 Audio Generation")
        
        with st.spinner("Generating audio..."):
            audio_data = text2audio(user_review)
        
        # 显示音频播放器
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        
        st.audio(audio_array, sample_rate=sample_rate)
        
        # 添加一个清除按钮
        if st.button("Clear Results"):
            st.session_state.show_results = False
            st.session_state.current_review = ""
            st.rerun()

if __name__ == "__main__":
    main()
