import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import torch
from PIL import Image
from io import BytesIO

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit Title
st.title("Image Captioning with Text-to-Audio")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate and Display Caption
    st.write("Generating caption...")
    caption = generate_caption(image)
    st.write(f"**Caption**: {caption}")

    # Text-to-Audio Conversion
    st.write("Convert the caption to audio:")
    language = st.selectbox("Select language:", ["en", "es", "fr", "de", "hi"])
    
    if st.button("Generate Audio"):
        if caption.strip():
            try:
                # Generate audio using gTTS
                tts = gTTS(text=caption, lang=language)
                
                # Save to a BytesIO object
                audio_file = BytesIO()
                tts.write_to_fp(audio_file)
                audio_file.seek(0)

                # Play audio and provide download option
                st.audio(audio_file, format="audio/mp3")
                st.download_button(
                    "Download Audio",
                    audio_file,
                    file_name="caption_audio.mp3",
                    mime="audio/mp3",
                )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Caption is empty. Please upload an image first.")
