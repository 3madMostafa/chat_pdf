import streamlit as st
import os
import re
import requests
import google.generativeai as genai
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
import pytesseract
from PIL import Image
from gtts import gTTS
from io import BytesIO
from pdf2image import convert_from_bytes
from langdetect import detect, DetectorFactory
import time
from datetime import datetime

# Ensure consistent language detection
DetectorFactory.seed = 0

# ===================================================================
# CONFIGURATION & INITIALIZATION
# ===================================================================

class Config:
    """Application configuration and constants"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDOwMCTh0tc1Qy5bkTFaaEb262Mv6A5XRU"
    MODEL_NAME = "models/gemini-1.5-pro"
    SUPPORTED_FILE_TYPES = ["pdf", "txt", "jpg", "jpeg", "png"]
    MAX_PREVIEW_LENGTH = 500
    DEFAULT_LANGUAGE = "en"

# Initialize Gemini AI
genai.configure(api_key=Config.GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name=Config.MODEL_NAME)

# ===================================================================
# CORE AI PROCESSING FUNCTIONS
# ===================================================================

class AIProcessor:
    """Advanced AI processing with Gemini integration"""
    
    @staticmethod
    def generate_response(prompt: str, temperature: float = 0.7) -> str:
        """
        Generate AI response with enhanced error handling and streaming capability
        
        Args:
            prompt: Input prompt for the AI model
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            Generated text response
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.candidates[0].content.parts[0].text
        
        except Exception as e:
            st.error(f"AI Processing Error: {str(e)}")
            return "An error occurred while processing your request. Please try again."
    
    @staticmethod
    def generate_streaming_response(prompt: str):
        """Generate streaming response for real-time display"""
        try:
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Streaming Error: {str(e)}"

# ===================================================================
# DOCUMENT PROCESSING UTILITIES
# ===================================================================

class DocumentProcessor:
    """Advanced document processing with OCR and text extraction"""
    
    @staticmethod
    def extract_pdf_text(pdf_file) -> str:
        """
        Extract text from PDF using PyPDF2, fallback to OCR for scanned documents
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            
            # Extract text from each page
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Fallback to OCR if no text extracted (scanned PDF)
            if not text.strip():
                with st.spinner("Processing scanned document with OCR..."):
                    images = convert_from_bytes(pdf_bytes)
                    for idx, image in enumerate(images):
                        page_text = pytesseract.image_to_string(image, lang="ara+eng")
                        text += f"\n--- Page {idx + 1} ---\n{page_text}\n"
            
            return text.strip()
        
        except Exception as e:
            st.error(f"PDF Processing Error: {str(e)}")
            return ""
    
    @staticmethod
    def extract_image_text(image_file) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image_file: Uploaded image file object
            
        Returns:
            Extracted text content
        """
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image, lang="ara+eng")
            return text.strip()
        except Exception as e:
            st.error(f"Image Processing Error: {str(e)}")
            return ""
    
    @staticmethod
    def process_file(uploaded_file) -> str:
        """
        Universal file processor supporting PDF, TXT, and images
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content
        """
        if uploaded_file is None:
            return ""
        
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return DocumentProcessor.extract_pdf_text(uploaded_file)
        
        elif file_type in ["image/jpeg", "image/jpg", "image/png"]:
            return DocumentProcessor.extract_image_text(uploaded_file)
        
        elif file_type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        
        else:
            try:
                return uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"Unsupported file format: {str(e)}")
                return ""

# ===================================================================
# LANGUAGE & MEDIA UTILITIES
# ===================================================================

class LanguageUtils:
    """Language detection and text-to-speech utilities"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language of input text"""
        try:
            lang = detect(text)
            return "ar" if lang.startswith("ar") else "en"
        except:
            return Config.DEFAULT_LANGUAGE
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Convert language code to full name"""
        return "Arabic" if lang_code == "ar" else "English"

class MediaProcessor:
    """Media processing for audio and video content"""
    
    @staticmethod
    def extract_youtube_transcript(video_url: str) -> str:
        """
        Extract transcript from YouTube video
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Video transcript text
        """
        try:
            video_id_match = re.search(r"v=([^&]+)", video_url)
            video_id = video_id_match.group(1) if video_id_match else video_url.split("/")[-1]
            
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'ar']
            )
            transcript = " ".join([item["text"] for item in transcript_list])
            return transcript
        
        except Exception as e:
            return f"Unable to extract transcript from video. Error: {str(e)}"
    
    @staticmethod
    def generate_audio(text: str) -> BytesIO:
        """
        Generate text-to-speech audio
        
        Args:
            text: Input text to convert
            
        Returns:
            Audio file as BytesIO object
        """
        try:
            detected_lang = LanguageUtils.detect_language(text)
            tts = gTTS(text, lang=detected_lang)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            return audio_bytes
        except Exception as e:
            st.error(f"Audio Generation Error: {str(e)}")
            return None

# ===================================================================
# FEATURE IMPLEMENTATIONS
# ===================================================================

class DocumentChat:
    """Interactive document Q&A system"""
    
    @staticmethod
    def render():
        st.header("Document Intelligence")
        st.markdown("Upload any document and engage in an intelligent conversation about its content")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=Config.SUPPORTED_FILE_TYPES,
            help="Supports PDF, TXT, and image files"
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                content = DocumentProcessor.process_file(uploaded_file)
            
            if not content.strip():
                st.error("No content could be extracted from the document")
                return
            
            detected_lang = LanguageUtils.detect_language(content)
            lang_name = LanguageUtils.get_language_name(detected_lang)
            
            st.success(f"Document processed successfully | Language: {lang_name}")
            
            # Generate AI-powered summary
            with st.expander("Document Summary"):
                summary_prompt = f"""Analyze the following document and provide a comprehensive summary.
                Use clear, professional language in {lang_name}.
                
                Document Content:
                {content}
                
                Provide:
                1. Main topic and purpose
                2. Key points (3-5 bullet points)
                3. Important insights or conclusions"""
                
                with st.spinner("Generating intelligent summary..."):
                    summary = AIProcessor.generate_response(summary_prompt)
                    st.markdown(summary)
            
            # Q&A Interface
            st.subheader("Ask Questions")
            question = st.text_input("What would you like to know about this document?")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("Submit", type="primary")
            
            if ask_button and question:
                qa_prompt = f"""Based on the following document, answer the question comprehensively and accurately.
                
                Document Content:
                {content}
                
                Question: {question}
                
                Provide a detailed, well-structured answer in {lang_name}. Include relevant quotes or references from the document when appropriate."""
                
                with st.spinner("Analyzing and generating response..."):
                    answer = AIProcessor.generate_response(qa_prompt)
                    
                    st.markdown("### Response")
                    st.info(answer)

class QuizGenerator:
    """AI-powered quiz generation system"""
    
    @staticmethod
    def render():
        st.header("Intelligent Quiz Generator")
        st.markdown("Transform any content into comprehensive assessment questions")
        
        input_method = st.radio(
            "Content Source",
            ["Upload File", "Paste Text"],
            horizontal=True
        )
        
        content = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Select Document",
                type=Config.SUPPORTED_FILE_TYPES,
                key="quiz_file"
            )
            if uploaded_file:
                with st.spinner("Extracting content..."):
                    content = DocumentProcessor.process_file(uploaded_file)
        else:
            content = st.text_area(
                "Enter Text Content",
                height=300,
                placeholder="Paste your educational content here..."
            )
        
        if content:
            preview = content[:Config.MAX_PREVIEW_LENGTH]
            preview += "..." if len(content) > Config.MAX_PREVIEW_LENGTH else ""
            
            with st.expander("Content Preview"):
                st.text(preview)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_questions = st.number_input("Number of Questions", 1, 20, 5)
            with col2:
                difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
            with col3:
                question_type = st.selectbox("Question Type", ["Multiple Choice", "True/False", "Mixed"])
            
            if st.button("Generate Quiz", type="primary"):
                detected_lang = LanguageUtils.detect_language(content)
                lang_name = LanguageUtils.get_language_name(detected_lang)
                
                quiz_prompt = f"""Create a {difficulty.lower()} difficulty quiz with {num_questions} {question_type.lower()} questions based on the following content.
                
                Content:
                {content}
                
                Requirements:
                1. Each question should test understanding of key concepts
                2. For multiple choice: provide 4 options (A, B, C, D)
                3. Clearly mark the correct answer
                4. Include brief explanations for correct answers
                5. Use {lang_name} language
                6. Format professionally with clear numbering
                
                Output format:
                Question 1: [Question text]
                A) [Option]
                B) [Option]
                C) [Option]
                D) [Option]
                Correct Answer: [Letter]
                Explanation: [Brief explanation]
                """
                
                with st.spinner("Generating intelligent assessment..."):
                    quiz_text = AIProcessor.generate_response(quiz_prompt, temperature=0.8)
                    
                    st.markdown("### Generated Quiz")
                    st.markdown(quiz_text)
                    
                    # Download option
                    st.download_button(
                        "Download Quiz",
                        quiz_text,
                        file_name=f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

class YouTubeSummarizer:
    """AI-powered video content summarizer"""
    
    @staticmethod
    def render():
        st.header("Video Intelligence Analyzer")
        st.markdown("Extract and analyze educational content from YouTube videos")
        
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("Analyze", type="primary")
        
        if analyze_button and video_url:
            with st.spinner("Extracting video transcript..."):
                transcript = MediaProcessor.extract_youtube_transcript(video_url)
            
            if "Error" in transcript or "Unable" in transcript:
                st.error(transcript)
                return
            
            detected_lang = LanguageUtils.detect_language(transcript)
            lang_name = LanguageUtils.get_language_name(detected_lang)
            
            summary_prompt = f"""Analyze this educational video transcript and provide a comprehensive breakdown.
            
            Transcript:
            {transcript}
            
            Provide in {lang_name}:
            1. Executive Summary (2-3 sentences)
            2. Main Topics Covered (bullet points)
            3. Key Insights and Learning Points
            4. Practical Applications or Examples
            5. Conclusion and Takeaways
            
            Use professional, clear language suitable for educational purposes."""
            
            with st.spinner("Generating intelligent analysis..."):
                summary = AIProcessor.generate_response(summary_prompt)
                
                st.markdown("### Video Analysis")
                st.success(summary)
                
                # Additional features
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Key Concepts"):
                        concept_prompt = f"Extract and explain the top 5 key concepts from this content in {lang_name}:\n\n{transcript}"
                        concepts = AIProcessor.generate_response(concept_prompt)
                        st.info(concepts)
                
                with col2:
                    if st.button("Create Study Notes"):
                        notes_prompt = f"Create structured study notes from this content in {lang_name}:\n\n{transcript}"
                        notes = AIProcessor.generate_response(notes_prompt)
                        st.info(notes)

class ConversationalAI:
    """Advanced conversational AI assistant"""
    
    @staticmethod
    def render():
        st.header("AI Learning Assistant")
        st.markdown("Your intelligent companion for academic questions and discussions")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_query = st.chat_input("Ask me anything about your studies...")
        
        if user_query:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Generate AI response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                context_prompt = f"""You are an expert educational AI assistant helping students with their academic queries.
                
                Student Question: {user_query}
                
                Provide a comprehensive, well-structured response that:
                1. Directly answers the question
                2. Provides relevant examples or explanations
                3. Suggests related topics for deeper learning
                4. Uses clear, educational language
                
                Be helpful, accurate, and encouraging."""
                
                # Simulate streaming
                response = AIProcessor.generate_response(context_prompt)
                
                # Display with typing effect
                for char in response:
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

class FlashcardGenerator:
    """Intelligent flashcard creation system"""
    
    @staticmethod
    def render():
        st.header("Smart Flashcard Creator")
        st.markdown("Generate effective study flashcards from any content")
        
        input_method = st.radio(
            "Content Source",
            ["Upload File", "Paste Text"],
            horizontal=True,
            key="flashcard_input"
        )
        
        content = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Select Document",
                type=Config.SUPPORTED_FILE_TYPES,
                key="flashcard_file"
            )
            if uploaded_file:
                content = DocumentProcessor.process_file(uploaded_file)
        else:
            content = st.text_area(
                "Enter Content",
                height=300,
                placeholder="Paste content to convert into flashcards...",
                key="flashcard_text"
            )
        
        if content:
            num_cards = st.slider("Number of Flashcards", 5, 30, 10)
            
            if st.button("Generate Flashcards", type="primary"):
                detected_lang = LanguageUtils.detect_language(content)
                lang_name = LanguageUtils.get_language_name(detected_lang)
                
                flashcard_prompt = f"""Create {num_cards} high-quality flashcards from the following content.
                
                Content:
                {content}
                
                Requirements:
                1. Each flashcard should have a clear, concise question
                2. Provide accurate, comprehensive answers
                3. Cover the most important concepts
                4. Use {lang_name} language
                5. Format as: "Card X: Question: ... | Answer: ..."
                
                Focus on key concepts, definitions, and important relationships."""
                
                with st.spinner("Creating intelligent flashcards..."):
                    flashcards = AIProcessor.generate_response(flashcard_prompt)
                    
                    # Parse and display flashcards
                    cards = re.split(r'Card\s+\d+:', flashcards)
                    
                    for idx, card in enumerate(cards[1:], 1):
                        with st.expander(f"Flashcard {idx}"):
                            parts = card.split('|')
                            if len(parts) >= 2:
                                question = parts[0].replace("Question:", "").strip()
                                answer = parts[1].replace("Answer:", "").strip()
                                
                                st.markdown(f"**Question:** {question}")
                                st.markdown(f"**Answer:** {answer}")
                            else:
                                st.markdown(card.strip())

class VoiceAssistant:
    """Text-to-speech AI assistant"""
    
    @staticmethod
    def render():
        st.header("Voice Learning Assistant")
        st.markdown("Convert text content to natural speech for audio learning")
        
        input_method = st.radio(
            "Content Source",
            ["Upload File", "Enter Text"],
            horizontal=True
        )
        
        content = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Select Document",
                type=Config.SUPPORTED_FILE_TYPES
            )
            if uploaded_file:
                content = DocumentProcessor.process_file(uploaded_file)
        else:
            content = st.text_area(
                "Enter Text",
                height=300,
                placeholder="Enter text to convert to speech..."
            )
        
        if content.strip():
            detected_lang = LanguageUtils.detect_language(content)
            lang_name = LanguageUtils.get_language_name(detected_lang)
            
            st.info(f"Detected Language: {lang_name}")
            
            # AI Enhancement
            enhance = st.checkbox("Enhance text with AI for better clarity")
            
            if enhance:
                enhancement_prompt = f"""Improve and clarify this text for audio narration. Make it more conversational and easier to understand when spoken aloud.
                
                Original Text:
                {content}
                
                Provide the enhanced version in {lang_name}."""
                
                with st.spinner("Enhancing text with AI..."):
                    enhanced_text = AIProcessor.generate_response(enhancement_prompt)
                    
                    st.markdown("### Enhanced Text")
                    st.write(enhanced_text)
                    content = enhanced_text
            
            if st.button("Generate Audio", type="primary"):
                with st.spinner("Converting text to speech..."):
                    audio = MediaProcessor.generate_audio(content)
                    
                    if audio:
                        st.success("Audio generated successfully")
                        st.audio(audio, format="audio/mp3")
                        
                        st.download_button(
                            "Download Audio",
                            audio,
                            file_name=f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                            mime="audio/mp3"
                        )

class ImageAnalyzer:
    """AI-powered image content analysis"""
    
    @staticmethod
    def render():
        st.header("Visual Content Analyzer")
        st.markdown("Extract and analyze text from images using advanced OCR and AI")
        
        image_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing text or diagrams"
        )
        
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("Analyzing image content..."):
                extracted_text = DocumentProcessor.extract_image_text(image_file)
            
            if extracted_text.strip():
                st.success("Text extracted successfully")
                
                with st.expander("Extracted Text"):
                    st.text(extracted_text)
                
                detected_lang = LanguageUtils.detect_language(extracted_text)
                lang_name = LanguageUtils.get_language_name(detected_lang)
                
                user_question = st.text_input("Ask a question about the image content")
                
                if st.button("Analyze", type="primary") and user_question:
                    analysis_prompt = f"""Analyze the following text extracted from an image and answer the question.
                    
                    Extracted Content:
                    {extracted_text}
                    
                    Question: {user_question}
                    
                    Provide a comprehensive answer in {lang_name}, including:
                    1. Direct answer to the question
                    2. Relevant context from the image
                    3. Additional insights if applicable"""
                    
                    with st.spinner("Generating intelligent response..."):
                        answer = AIProcessor.generate_response(analysis_prompt)
                        
                        st.markdown("### Analysis Result")
                        st.success(answer)
            else:
                st.warning("No text could be extracted from this image")

# ===================================================================
# APPLICATION MAIN INTERFACE
# ===================================================================

def apply_custom_styling():
    """Apply custom CSS for professional appearance"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
            font-weight: 500;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 5px;
        }
        h1 {
            color: #1E3A8A;
            font-weight: 700;
            padding-bottom: 1rem;
            border-bottom: 3px solid #3B82F6;
        }
        h2 {
            color: #1E40AF;
            font-weight: 600;
        }
        .stAlert {
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Educational Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_styling()
    
    # Header
    st.title("Advanced AI Educational Assistant")
    st.markdown("Powered by Google Gemini 1.5 Pro | Professional Learning Support System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("Select a feature below to get started")
    
    features = {
        "Document Intelligence": DocumentChat,
        "Quiz Generator": QuizGenerator,
        "Video Analyzer": YouTubeSummarizer,
        "AI Assistant": ConversationalAI,
        "Flashcard Creator": FlashcardGenerator,
        "Voice Assistant": VoiceAssistant,
        "Image Analyzer": ImageAnalyzer
    }
    
    selected_feature = st.sidebar.radio(
        "Features",
        list(features.keys()),
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Information panel
    with st.sidebar.expander("About This Application"):
        st.markdown("""
        This professional AI assistant leverages advanced machine learning 
        to provide comprehensive educational support.
        
        **Capabilities:**
        - Intelligent document analysis
        - Automated quiz generation
        - Video content summarization
        - Interactive Q&A
        - Flashcard creation
        - Text-to-speech conversion
        - Image text extraction
        
        **Technology Stack:**
        - Google Gemini 1.5 Pro
        - Advanced OCR Processing
        - Natural Language Understanding
        - Multi-language Support
        """)
    
    # Render selected feature
    st.markdown("---")
    selected_class = features[selected_feature]
    selected_class.render()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Version 2.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}")
    st.sidebar.caption("Developed with Google Gemini AI")

if __name__ == "__main__":
    main()

