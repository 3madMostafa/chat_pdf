<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=180&section=header&text=AI%20Study%20Assistant&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Chat%20with%20your%20files%2C%20make%20quizzes%20and%20flashcards&descAlignY=60&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=36BCF7&center=true&vCenter=true&width=600&lines=Upload+a+PDF+and+ask+questions;Make+a+quiz+in+one+click;Summarize+any+YouTube+video;Works+in+Arabic+and+English" alt="Typing SVG"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tesseract%20OCR-3E8E41?style=for-the-badge"/>
</p>

## What is this?

A study helper app built with **Streamlit** and **Google Gemini**.
You upload your study files (PDF, image or text) and the app helps you learn from them.

It also reads **scanned PDFs and images** with OCR, in **Arabic and English**.

## Features

| Tool | What it does |
|------|--------------|
| 📄 **Document Intelligence** | Upload a file and ask questions about it |
| 📝 **Quiz Generator** | Makes a quiz from your file |
| 🎬 **Video Analyzer** | Paste a YouTube link and get a summary of the video |
| 🤖 **AI Learning Assistant** | Ask any study question |
| 🃏 **Flashcard Creator** | Makes flashcards for quick review |
| 🔊 **Voice Assistant** | Reads the answer out loud |
| 🖼️ **Visual Analyzer** | Reads text from images |

## How it works

```mermaid
flowchart LR
    A[PDF / image / text] --> B{Has text?}
    B -- yes --> C[Read text]
    B -- no, scanned --> D[OCR: Arabic + English]
    C --> E[Gemini]
    D --> E
    F[YouTube link] --> G[Get subtitles] --> E
    E --> H[Answer / quiz / flashcards]
    H --> I[Text to speech]
```

## How to run

**1. Install the tools**

- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). Add the Arabic language pack.
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases). This is needed to read scanned PDFs.

**2. Install the Python packages**

```bash
pip install -r requirements.txt
```

**3. Add your API key**

Copy `env.template` to a new file called `.env` and add your key:

```
GOOGLE_API_KEY=your-key-here
```

You can get a free key from [Google AI Studio](https://aistudio.google.com/apikey).

> ⚠️ Never upload your `.env` file. It is already in `.gitignore`.

**4. Start the app**

```bash
streamlit run app.py
```

## Built with

- **Streamlit**: the web app
- **Google Gemini**: answers, quizzes, summaries
- **PyPDF2 + pdf2image + Tesseract**: reading PDFs and images
- **youtube-transcript-api**: getting video subtitles
- **gTTS**: text to speech
- **langdetect**: finding the language of the text

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=100&section=footer" width="100%"/>
</p>
