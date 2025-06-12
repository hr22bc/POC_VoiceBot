import streamlit as st
from dotenv import load_dotenv
from utils import (
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

    input_mode = st.radio("Choose input method:", ["Type", "Speak"])
    user_question = ""

    # ----------------------------
    # Type Mode
    # ----------------------------
    if input_mode == "Type":
        user_question = st.text_input("Type your question", key="text_query")
        if st.button("🔍 Ask"):
            if not user_question:
                st.warning("Please provide a question.")
            else:
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

        audio_bytes = audio_recorder()
        if audio_bytes:
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
                            st.error(" Try Again !!! Could not understand the audio. Please speak more clearly.")
                        except sr.RequestError as e:
                            st.error(f"Speech recognition service error: {e}")

                    os.unlink(tmp_audio_path)
                except Exception as e:
                    st.error(f" Try Again !!Audio processing error: {e}")

    # ----------------------------
    # Chat History
    # ----------------------------
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🕑 Chat History")
        for turn in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {turn['query']}")
            st.markdown(f"**Bot:** {turn['response']}")
