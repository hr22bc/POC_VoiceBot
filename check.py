import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

import av
import numpy as np

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Just pass through
        return frame

st.title("Voice Input with Streamlit WebRTC")
webrtc_streamer(
    key="audio",
    mode="SENDONLY",
    in_audio_enabled=True,
    audio_processor_factory=AudioProcessor,
)
