import uuid
from io import BytesIO
from gtts import gTTS
from deep_translator import GoogleTranslator
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
import speech_recognition as sr
import tempfile
import re

# -------------------------
# Document + Embedding Utils
# -------------------------

def load_document_vectorstore(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# -------------------------
# QA Chain + History Prompt
# -------------------------

def get_qa_response(query: str, retriever, history: list = None, target_lang: str = "en"):
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o")

    QA_PROMPT = """
    You are a helpful assistant answering user questions based only on the document context and chat history.
    Avoid using external knowledge. Do not guess if the context doesn't support the answer.

    Previous Conversation:
    {history_block}

    Question: {question}
    Context:
    {context}

    Answer:
    """

    friendly_phrases = {
        "hi", "hello", "hey", "how are you", "good morning", "good evening",
        "good afternoon", "can you help me", "who are you", "what do you do",
        "are you a bot", "nice to meet you", "thank you", "thanks", "bye"
    }

    def is_friendly_query(query: str):
        import re
        normalized = re.sub(r"[^\w\s]", "", query.strip().lower())
        return normalized in friendly_phrases

    # Translate query to English if needed
    if target_lang != "en":
        try:
            query_in_english = GoogleTranslator(source="auto", target='en').translate(query)
        except:
            query_in_english = query
    else:
        query_in_english = query

    if is_friendly_query(query_in_english):
        response_text = "Hello! How can I assist you today?"
        if target_lang != "en":
            response_text = GoogleTranslator(source='en', target=target_lang).translate(response_text)
        return response_text, "", ""

    # Translate chat history if needed
    history_block = ""
    if history:
        for turn in history[-5:]:
            q = turn.get("query", "").strip()
            a = turn.get("response", "").strip()
            if q and a and not is_friendly_query(q):
                if target_lang != "en":
                    try:
                        q = GoogleTranslator(source="auto", target='en').translate(q)
                        a = GoogleTranslator(source="auto", target='en').translate(a)
                    except:
                        pass
                history_block += f"User: {q}\nAssistant: {a}\n"

    docs = retriever.get_relevant_documents(query_in_english)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = QA_PROMPT.format(
        history_block=history_block.strip(),
        question=query_in_english,
        context=context_text
    )

    answer = llm.invoke(prompt).content.strip()

    # Translate answer back to target language
    if target_lang != "en":
        try:
            translated_answer = GoogleTranslator(source='en', target=target_lang).translate(answer)
        except:
            translated_answer = answer
    else:
        translated_answer = answer

    return translated_answer, answer, context_text

# -------------------------
# Language Utilities
# -------------------------

def translate_text(text: str, target_lang: str):
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

def get_language_code(language_name: str):
    lang_map = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
        "German": "de", "Gujarati": "gu", "Tamil": "ta", "Telugu": "te",
        "Thai": "th", "Arabic": "ar"
    }
    return lang_map.get(language_name, "en")

# -------------------------
# Audio Utilities
# -------------------------

def text_to_audio(text: str, lang_code: str):
    tts = gTTS(text=text, lang=lang_code)
    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return audio

def transcribe_audio_file(audio_path: str, language_code: str = "en"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio_data, language=language_code)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# -------------------------
# Misc Utilities
# -------------------------

def generate_session_id():
    return str(uuid.uuid4())[:8]
