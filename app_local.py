import streamlit as st
from dotenv import load_dotenv
from utils_local import (
    load_document_vectorstore,
    get_qa_response,
    translate_text,
    get_language_code,
    text_to_audio,
    generate_session_id
)
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr

# ----------------------------
# Page Configuration
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ----------------------------
# Login System
# ----------------------------
def login():
    st.title("🔐 Login to Voice Chatbot")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    valid_users = {
        "admin": "admin123",
        "user1": "pass1",
        "bhavna": "1234"
    }

    if login_btn:
        if username in valid_users and valid_users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.session_id = generate_session_id()
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Try again.")

# ----------------------------
# Session Check
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ----------------------------
# Show Login or App
# ----------------------------
if not st.session_state.logged_in:
    login()
    st.stop()
###########################3
st.set_page_config(page_title="Multilingual Voice Chatbot", layout="centered")
st.title("Multilingual Voice Chatbot")
st.markdown("Ask questions based on uploaded documents in your selected language. Get responses in both text and audio.")

# ----------------------------
# Language Selection
# ----------------------------
lang_options = ["English", "Hindi", "Thai", "Spanish", "Arabic"]
language = st.selectbox("🌐 Select your language", lang_options)
lang_code = get_language_code(language)

# ----------------------------
# Upload Document
# ----------------------------
uploaded_file = st.file_uploader("📄 Upload a document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
        tmp.write(uploaded_file.read())
        doc_path = tmp.name

    st.success("✅ Document uploaded successfully!")
    retriever = load_document_vectorstore(doc_path)

    # ----------------------------
    # Conversation Section
    # ----------------------------
    st.subheader("💬 Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if "is_processing_audio" not in st.session_state:
        st.session_state.is_processing_audio = False

    input_mode = st.radio("Choose input method:", ["Type", "Speak"])
    user_question = ""

    # ----------------------------
    # Type Mode
    # ----------------------------
    if input_mode == "Type":
        user_question = st.text_input("Type your question", key="text_query")
        ask_button = st.button("🔍 Ask", disabled=not user_question.strip())
        if ask_button:
            answer, _, _ = get_qa_response(user_question, retriever, st.session_state.chat_history)
            translated_answer = translate_text(answer, lang_code)
            st.session_state.chat_history.append({"query": user_question, "response": translated_answer})
            st.markdown(f"**🧠 Bot Response:** {translated_answer}")
            audio_data = text_to_audio(translated_answer, lang_code)
            st.audio(audio_data, format="audio/mp3")

    # ----------------------------
    # Speak Mode
    # ----------------------------
    elif input_mode == "Speak":
        st.markdown("### 🎙️ Voice Recording")
        st.markdown("**Instructions:** Click the microphone icon. When it turns **red**, start speaking. Click again to **stop** recording. and it will trun **Black**. **First CLick is to setup Microphone and Testing**. ")
        
        if st.session_state.is_processing_audio:
            st.warning("⏳ Please wait while your last recording is being processed...")
        else:
            audio_bytes = audio_recorder(pause_threshold=4.0)
            if audio_bytes:
                st.session_state.is_processing_audio = True
                st.audio(audio_bytes, format="audio/wav")

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    tmp_audio.write(audio_bytes)
                    tmp_audio_path = tmp_audio.name

                with st.spinner("🔄 Processing..."):
                    try:
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(tmp_audio_path) as source:
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            audio_data = recognizer.record(source)

                            try:
                                user_question = recognizer.recognize_google(audio_data, language=lang_code)
                                st.success(f"🗣️ **Recognized Text:** {user_question}")

                                answer, _, _ = get_qa_response(user_question, retriever, st.session_state.chat_history)
                                translated_answer = translate_text(answer, lang_code)
                                st.session_state.chat_history.append({"query": user_question, "response": translated_answer})
                                st.markdown(f"**🧠 Bot Response:** {translated_answer}")
                                audio_data = text_to_audio(translated_answer, lang_code)
                                st.audio(audio_data, format="audio/mp3")

                            except sr.UnknownValueError:
                                st.error(" Test done !!! Microphone is Stable. Please Speak. or Try Again as no audio recorded")
                            except sr.RequestError as e:
                                st.error(f"Speech recognition service error: {e}")

                        os.unlink(tmp_audio_path)
                    except Exception as e:
                        st.error(f" Try Again !!Audio processing error: {e}")
                    finally:
                        st.session_state.is_processing_audio = False

    # ----------------------------
    # Chat History
    # ----------------------------
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🕑 Chat History")
        for turn in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {turn['query']}")
            st.markdown(f"**Bot:** {turn['response']}")

# Logout Option
# ----------------------------
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
if st.button("User Logout"):
    st.session_state.logged_in = False
    st.session_state.chat_history = []
    st.rerun()