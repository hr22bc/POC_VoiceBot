
import streamlit as st
from dotenv import load_dotenv
from utils import (
    load_document_vectorstore,
    get_qa_response,
    translate_text,
    get_language_code,
    text_to_audio,
    transcribe_audio_file,
    generate_session_id
)
import tempfile
import os
import threading
import wave
import pyaudio
import speech_recognition as sr

# ----------------------------
# Audio Recording Class
# ----------------------------
class AudioRecorder:
    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()

    def start_recording(self):
        self.frames = []
        self.recording = True

        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            while self.recording:
                data = self.stream.read(self.chunk)
                self.frames.append(data)

        except Exception as e:
            st.error(f"Recording error: {e}")

        finally:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()

    def stop_recording(self):
        self.recording = False

    def save_recording(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def close(self):
        self.audio.terminate()

# ----------------------------
# Page Configuration
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Multilingual Voice Chatbot", layout="centered")
st.title("🌍 Multilingual Voice Chatbot")
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

    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()

    if 'recording_thread' not in st.session_state:
        st.session_state.recording_thread = None

    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

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

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔴 START Recording") and not st.session_state.is_recording:
                st.session_state.is_recording = True
                st.session_state.recording_thread = threading.Thread(target=st.session_state.recorder.start_recording)
                st.session_state.recording_thread.start()
                st.rerun()

        with col2:
            if st.button("⏹️ STOP Recording") and st.session_state.is_recording:
                st.session_state.recorder.stop_recording()
                st.session_state.is_recording = False
                if st.session_state.recording_thread:
                    st.session_state.recording_thread.join()
                st.rerun()

        with col3:
            if st.button("🗑️ Clear"):
                if st.session_state.is_recording:
                    st.session_state.recorder.stop_recording()
                    st.session_state.is_recording = False
                st.session_state.recorder = AudioRecorder()
                st.rerun()

        if st.session_state.is_recording:
            st.success("🔴 Recording... Speak now!")
            st.info("Click STOP when you're finished speaking")
        elif st.session_state.recorder.frames:
            st.info("⏸️ Recording stopped. Click 'Send Query' to process.")

            if st.button("📤 Send Query"):
                with st.spinner("🔄 Converting speech to text..."):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            st.session_state.recorder.save_recording(tmp_file.name)
                            temp_audio_path = tmp_file.name

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_audio_path) as source:
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            audio_data = recognizer.record(source)
                            try:
                                user_question = recognizer.recognize_google(audio_data, language=lang_code)
                                st.success(f"🗣️ **Recognized Text:** {user_question}")

                                # LLM response generation
                                answer, _, _ = get_qa_response(user_question, retriever, st.session_state.chat_history)
                                translated_answer = translate_text(answer, lang_code)
                                st.session_state.chat_history.append({"query": user_question, "response": translated_answer})
                                st.markdown(f"**🧠 Bot Response:** {translated_answer}")
                                audio_data = text_to_audio(translated_answer, lang_code)
                                st.audio(audio_data, format="audio/mp3")

                            except sr.UnknownValueError:
                                st.error("❌ Could not understand the audio. Please speak more clearly.")
                            except sr.RequestError as e:
                                st.error(f"❌ Speech recognition service error: {e}")

                        os.unlink(temp_audio_path)
                    except Exception as e:
                        st.error(f"❌ Audio processing error: {e}")

    # ----------------------------
    # Chat History
    # ----------------------------
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🕑 Chat History")
        for turn in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {turn['query']}")
            st.markdown(f"**Bot:** {turn['response']}")
