import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import torch
from PIL import Image
from io import BytesIO
#----------------------captioner imports----------
import json
from decoder import Decoder
from encoder import Encoder
from dataset import pil_loader
from train import data_transforms
#-------------------------------------------------

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Function to generate captions
def generate_caption(img):
    # initiaziling vocab
    word_dict = json.load(open('word_dict.json', 'r'))
    vocabulary_size = len(word_dict)
    encoder = Encoder(network='vgg19')
    decoder = Decoder(vocabulary_size, encoder.dim)
    decoder.load_state_dict(torch.load('model_10.pth', map_location=torch.device('cpu')))
    encoder.eval()
    decoder.eval()

    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    img_features = encoder(img)
    beam_size=3
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens=[]
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break
    caption = " ".join(sentence_tokens[1:-1])
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
    # caption = "पृष्ठभूमि में एक पहाड़ और एक बड़े नीले आकाश के साथ एक नदी का दृश्य"
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
