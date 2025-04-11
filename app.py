import streamlit as st
import os
import re
import requests
import google.generativeai as genai
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
#import chromadb
#from chromadb.utils import embedding_functions
import pytesseract
from PIL import Image
from gtts import gTTS
from io import BytesIO

# مكتبة إضافية لتحويل صفحات PDF إلى صور (تتطلب تثبيت poppler)
from pdf2image import convert_from_bytes

# إعداد langdetect مع seed لضمان ثبات النتائج
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# -------------------------------------------------------------------
# إعداد Gemini API
# -------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDtwKF1t3fcooCHNauZoMk35h5jAHHLKbs"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# -------------------------------------------------------------------
# إعداد ChromaDB لاسترجاع السياق (مجموعة المنهج)
# -------------------------------------------------------------------
#chromadb_client = chromadb.Client()
#embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
#collection_names = [col.name for col in chromadb_client.list_collections()]
#if "curriculum" in collection_names:
#    collection = chromadb_client.get_collection("curriculum")
#else:
#    collection = chromadb_client.create_collection(name="curriculum", embedding_function=embedding_function)

# -------------------------------------------------------------------
# دوال مساعدة
# -------------------------------------------------------------------

#def augment_prompt_with_chroma(prompt):
#    results = collection.query(query_texts=[prompt], n_results=3)
#    context = ""
#    if results and results.get("documents") and results["documents"][0]:
#        context = " ".join(results["documents"][0])
#    augmented = prompt + "\n\nRelevant Context:\n" + context
#    return augmented

def extract_pdf_text(pdf_file):
    """
    محاولة استخراج النص باستخدام PyPDF2، وفي حال عدم وجود نص (ملف مسح ضوئي)
    نقوم بتحويل الصفحات إلى صور ثم استخدام OCR.
    """
    text = ""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # إذا لم يستخرج النص، نفترض أن الملف عبارة عن صور (مسح ضوئي)
        if text.strip() == "":
            images = convert_from_bytes(pdf_bytes)
            for image in images:
                page_text = pytesseract.image_to_string(image, lang="ara+eng")
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"خطأ في معالجة PDF: {e}")
        return ""

def extract_youtube_transcript(video_url):
    video_id_match = re.search(r"v=([^&]+)", video_url)
    video_id = video_id_match.group(1) if video_id_match else video_url.split("/")[-1]
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ar'])
        transcript = " ".join([item["text"] for item in transcript_list])
        return transcript
    except Exception as e:
        return f"❌ لم يتم العثور على تفريغ لهذا الفيديو باللغة العربية. الخطأ: {e}"

def detect_language(text):
    try:
        lang = detect(text)
        return "ar" if lang.startswith("ar") else "en"
    except Exception:
        return "ar"  # بديل افتراضي

def get_tts_audio(text):
    try:
        detected_lang = detect_language(text)
        tts = gTTS(text, lang=detected_lang)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.error(f"خطأ في تحويل النص إلى صوت: {e}")
        return None

def process_uploaded_file(uploaded_file):
    """
    يدعم رفع الملفات لأنواع:
    - PDF: استخراج النص باستخدام PyPDF2، وإذا لم يكن هناك نص (مسح ضوئي)
      يقوم بتحويل الصفحات إلى صور واستخدام OCR.
    - TXT: قراءة النص مباشرة.
    - الصور (JPG, JPEG, PNG): استخدام OCR لقراءة النص من الصورة.
    """
    content = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            content = extract_pdf_text(uploaded_file)
        elif uploaded_file.type in ["image/jpeg", "image/jpg", "image/png"]:
            image = Image.open(uploaded_file)
            content = pytesseract.image_to_string(image, lang="ara+eng")
        elif uploaded_file.type in ["text/plain"]:
            content = uploaded_file.read().decode("utf-8")
        else:
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception as e:
                st.error(f"خطأ أثناء قراءة الملف: {e}")
    return content

def call_llm(prompt):
   # augmented_prompt = augment_prompt_with_chroma(prompt)
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

# -------------------------------------------------------------------
# الميزة 1: دردشة مع الملف
# -------------------------------------------------------------------
def chat_with_pdf_feature(language):
    st.subheader("📄 دردشة مع الملف")
    uploaded_file = st.file_uploader("ارفع ملف (PDF, TXT, صورة)", type=["pdf", "txt", "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        content = process_uploaded_file(uploaded_file)
        if content.strip() == "":
            st.error("لم يتم استخراج أي نص من الملف.")
            return
        detected_lang = detect_language(content)
        language_str = "العربية" if detected_lang == "ar" else "الإنجليزية"
        summary_prompt = f"الوثيقة التالية تحتوي على معلومات، لخص النص بشكل موجز باللغة {language_str}:\n\n{content}"
        summary = call_llm(summary_prompt)
        st.markdown("### الملخص")
        st.write(summary)
        question = st.text_input("اطرح سؤالك عن الملف")
        if st.button("إرسال السؤال"):
            if question:
                prompt = f"الوثيقة:\n{content}\n\nالسؤال: {question}\nالرجاء تقديم إجابة باللغة {language_str}."
                answer = call_llm(prompt)
                st.markdown("### الإجابة")
                st.write(answer)
            else:
                st.warning("الرجاء إدخال سؤال.")

# -------------------------------------------------------------------
# الميزة 2: مولّد الاختبارات
# -------------------------------------------------------------------
def quiz_generator_feature(language):
    st.subheader("🧠 مولّد الاختبارات")
    input_method = st.radio("اختر طريقة الإدخال", ("رفع ملف", "نسخ ولصق النص"))
    content = ""
    if input_method == "رفع ملف":
        uploaded_file = st.file_uploader("ارفع ملف (PDF, TXT, صورة)", type=["pdf", "txt", "jpg", "jpeg", "png"], key="quiz_file")
        if uploaded_file is not None:
            content = process_uploaded_file(uploaded_file)
    else:
        content = st.text_area("أدخل النص", height=300)
    
    if content:
        st.markdown("### معاينة المحتوى")
        st.write(content[:500] + "..." if len(content) > 500 else content)
        if st.button("توليد الاختبار"):
            detected_lang = detect_language(content)
            language_str = "العربية" if detected_lang == "ar" else "الإنجليزية"
            prompt = (
                f"أنشئ اختباراً متعدد الخيارات (MCQs) باللغة {language_str} قائم على المحتوى التالي. "
                "يجب أن يحتوي كل سؤال على أربع إجابات مع تحديد الإجابة الصحيحة أسفل كل سؤال.\n\n" + content
            )
            quiz_text = call_llm(prompt)
            st.markdown("### نتيجة الاختبار")
            quiz_blocks = quiz_text.strip().split("\n\n")
            for block in quiz_blocks:
                lines = block.strip().split("\n")
                if not lines:
                    continue
                st.markdown("---")
                for line in lines:
                    if line.startswith("السؤال") or line.startswith("س:") or "؟" in line:
                        st.markdown(f"**🟠 {line.strip()}**")
                    elif "الإجابة" in line:
                        st.markdown(f"✅ **{line.strip()}**")
                    elif re.match(r"^[أ-د]\)", line.strip()):
                        st.markdown(f"- {line.strip()}")
                    else:
                        st.write(line.strip())

# -------------------------------------------------------------------
# الميزة 3: تلخيص فيديو YouTube
# -------------------------------------------------------------------
def youtube_summarizer_feature(language):
    st.subheader("🎥 مولّد ملخص الفيديو من YouTube")
    video_url = st.text_input("أدخل رابط فيديو YouTube")
    if st.button("تلخيص الفيديو"):
        if video_url:
            transcript = extract_youtube_transcript(video_url)
            detected_lang = detect_language(transcript)
            language_str = "العربية" if detected_lang == "ar" else "الإنجليزية"
            prompt = (
                f"النص التالي هو تفريغ فيديو تعليمي. قم بتوليد ملخص شامل له باللغة {language_str}، دون عرض النص الأصلي:\n\n" + transcript
            )
            summary = call_llm(prompt)
            st.markdown("### الملخص")
            st.write(summary)
        else:
            st.warning("الرجاء إدخال رابط الفيديو.")

# -------------------------------------------------------------------
# الميزة 4: دردشة عامة (مدعومة بالمنهج)
# -------------------------------------------------------------------
def general_chat_feature(language):
    st.subheader("💡 دردشة عامة (مدعومة بالمنهج)")
    user_query = st.text_input("اكتب سؤالك هنا")
    if st.button("إرسال"):
        if user_query:
            school_keywords = ["مدرسة", "ثانوي", "امتحان", "تعليم", "منهج", "صف"]
           # if any(word in user_query for word in school_keywords):
           #     prompt = augment_prompt_with_chroma(user_query)
           # else:
            prompt = user_query
            answer = call_llm(prompt)
            st.markdown("### الإجابة")
            st.write(answer)
        else:
            st.warning("الرجاء كتابة سؤال.")

# -------------------------------------------------------------------
# الميزة 5: مولّد البطاقات التعليمية
# -------------------------------------------------------------------
def flashcard_generator_feature(language):
    st.subheader("🗂️ مولّد البطاقات التعليمية")
    input_method = st.radio("اختر طريقة الإدخال", ("رفع ملف", "نسخ ولصق النص"), key="flashcard_input")
    content = ""
    if input_method == "رفع ملف":
        uploaded_file = st.file_uploader("ارفع ملف (PDF, TXT, صورة)", type=["pdf", "txt", "jpg", "jpeg", "png"], key="flashcard_file")
        if uploaded_file is not None:
            content = process_uploaded_file(uploaded_file)
    else:
        content = st.text_area("أدخل النص", height=300, key="flashcard_text")
    
    if content:
        detected_lang = detect_language(content)
        language_str = "العربية" if detected_lang == "ar" else "الإنجليزية"
        prompt = (
            f"من النص التالي، أنشئ مجموعة من البطاقات التعليمية على هيئة سؤال-جواب. "
            f"يجب أن يكون كل سؤال وإجابته باللغة {language_str} وبصيغة موجزة:\n\n{content}\n\n"
            "يرجى إخراج البطاقات في الشكل التالي:\n"
            "البطاقة 1:\n"
            "السؤال: ...\n"
            "الإجابة: ...\n\n"
            "البطاقة 2: ..."
        )
        flashcards = call_llm(prompt)
        st.markdown("### البطاقات التعليمية")
        cards = re.split(r"(?:^|\n)(?=البطاقة\s*\d+[:：])", flashcards)
        for i, card in enumerate(cards):
            card = card.strip()
            if card:
                with st.expander(f"📘 بطاقة {i + 1}"):
                    lines = card.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("السؤال"):
                            st.markdown(f"**🟠 {line}**")
                        elif line.startswith("الإجابة"):
                            st.markdown(f"✅ {line}")
                        else:
                            st.write(line)

# -------------------------------------------------------------------
# الميزة 6: المساعد الصوتي (TTS مع تحسين النص عبر Gemini)
# -------------------------------------------------------------------
def arabic_voice_assistant_feature(language):
    st.subheader("🗣️ المساعد الصوتي")
    input_method = st.radio("اختر طريقة الإدخال", ("رفع ملف نصي", "كتابة النص يدويًا"))
    content = ""
    if input_method == "رفع ملف نصي":
        uploaded_file = st.file_uploader("ارفع ملف (PDF, TXT, صورة)", type=["pdf", "txt", "jpg", "jpeg", "png"])
        if uploaded_file is not None:
            content = process_uploaded_file(uploaded_file)
    else:
        content = st.text_area("اكتب النص هنا", height=300)
    if content.strip():
        detected_lang = detect_language(content)
        lang_name = "العربية" if detected_lang == "ar" else "الإنجليزية"
        st.success(f"✅ تم التعرف على اللغة: {lang_name}")
        # استخدام Gemini لإعادة صياغة وتحسين النص قبل تحويله إلى صوت
        if detected_lang == "ar":
            enhancement_prompt = f"قم بإعادة صياغة وتحسين النص التالي ليكون أكثر وضوحاً وفهماً:\n\n{content}"
        else:
            enhancement_prompt = f"Please rephrase and improve the following text to make it clearer and more understandable:\n\n{content}"
        enhanced_text = call_llm(enhancement_prompt)
        st.markdown("### النص المحسن")
        st.write(enhanced_text)
        audio = get_tts_audio(enhanced_text)
        st.audio(audio, format="audio/mp3")
    else:
        st.info("يرجى إدخال نص أو رفع ملف.")

# -------------------------------------------------------------------
# الميزة 7: السؤال من صورة
# -------------------------------------------------------------------
def question_from_image_feature(language):
    st.subheader("🖼️ اسأل عن محتوى صورة")
    image_file = st.file_uploader("ارفع صورة (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        extracted_text = pytesseract.image_to_string(image, lang="ara+eng")
        if extracted_text.strip():
            st.markdown("### النص المستخرج (تحضير للإجابة فقط)")
            st.write(extracted_text)
            user_question = st.text_input("❓ اسأل أي سؤال عن محتوى الصورة")
            if st.button("إرسال السؤال"):
                detected_lang = detect_language(extracted_text)
                language_str = "العربية" if detected_lang == "ar" else "الإنجليزية"
                prompt = (
                    f"الصورة تحتوي على النص التالي:\n\n{extracted_text}\n\n"
                    f"السؤال: {user_question}\nالرجاء الإجابة باللغة {language_str}."
                )
                answer = call_llm(prompt)
                st.markdown("### الإجابة")
                st.write(answer)
        else:
            st.warning("⚠️ لم يتم استخراج نص من الصورة.")

# -------------------------------------------------------------------
# تخطيط التطبيق الرئيسي والتوجيه عبر الشريط الجانبي
# -------------------------------------------------------------------
def main():
    st.title("المساعد التعليمي لطلبة الثانوية العامة المصرية (الثانوية العامة)")
    
    # تحسين تصميم الشريط الجانبي باستخدام أزرار الراديو وإضافة إيموجي لكل خاصية
    features = {
        "📄 دردشة مع الملف": chat_with_pdf_feature,
        "🧠 مولّد الاختبارات": quiz_generator_feature,
        "🎥 تلخيص فيديو YouTube": youtube_summarizer_feature,
        "💡 دردشة عامة (مدعومة بالمنهج)": general_chat_feature,
        "🗂️ مولّد البطاقات التعليمية": flashcard_generator_feature,
        "🗣️ المساعد الصوتي": arabic_voice_assistant_feature,
        "🖼️ سؤال من صورة": question_from_image_feature
    }
    feature_choice = st.sidebar.radio("اختر الخاصية", list(features.keys()))
    # اختيار لغة الواجهة (للنصوص غير المستخرجة من الملفات)
    language = st.sidebar.radio("اختر لغة الواجهة", ("Arabic", "English"))
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "هذا التطبيق مصمم لمساعدة طلاب الثانوية العامة بالمحتوى التعليمي. "
        "يستخدم واجهة Gemini 1.5 Pro مع دعم ChromaDB للبيانات المنهجية."
    )
    
    selected_function = features[feature_choice]
    selected_function(language)

if __name__ == "__main__":
    main()
