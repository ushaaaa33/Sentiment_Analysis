import re
import pickle
import numpy as np
import gradio as gr
import emoji
 
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
 
# 1. Load model & tokenizer 
MODEL_PATH     = "best_bilstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
 
print("Loading BiLSTM model …")
model = load_model(MODEL_PATH)
 
print("Loading tokenizer …")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
 
# 2. Constants (must match training exactly) 
MAX_LEN = 100   # maxlen=100 used in training pad_sequences
 
# 3. Lightweight preprocessing 
# Keep this minimal: the Tokenizer already lowercases and strips
# punctuation via its built-in filters. We only:
#   • convert emojis to text tokens  (e.g. 😊 → "smiling_face")
#   • strip non-ASCII foreign characters
# Do NOT over-clean — removing words reduces prediction signal.
 
def preprocess(raw_text: str) -> str:
    text = str(raw_text)
 
    # Convert emojis to readable words  (😊 → :smiling_face: → smiling_face)
    text = emoji.demojize(text)
    text = re.sub(r":([a-zA-Z0-9_]+):", r" \1 ", text)
 
    # Drop non-ASCII characters (foreign-language text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
 
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
 
    return text
 
 
# 4. Prediction function 
LABELS = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
 
def predict_sentiment(user_text: str):
    if not user_text or not user_text.strip():
        return "⚠️ Please enter some text.", {}
 
    # Preprocess
    cleaned = preprocess(user_text)
 
    if not cleaned.strip():
        return "⚠️ No recognisable text found after cleaning.", {}
 
    # Tokenise  (Tokenizer applies lowercase + punctuation filter internally)
    sequence = tokenizer.texts_to_sequences([cleaned])
 
    if not sequence or not sequence[0]:
        return "⚠️ None of the words were found in the vocabulary.", {}
 
    # ── KEY FIX: padding='pre' matches the training default ──
    padded = pad_sequences(sequence, maxlen=MAX_LEN)   # default padding='pre'
 
    # Predict
    probs      = model.predict(padded, verbose=0)[0]   # shape (3,)
    pred_class = int(np.argmax(probs))
 
    label      = LABELS[pred_class]
    confidence = {LABELS[i]: float(round(probs[i], 4)) for i in range(3)}
 
    return label, confidence
 
 
# 5. Gradio UI 
with gr.Blocks(title="Sentiment Analyser – BiLSTM") as demo:
 
    gr.Markdown(
        """
        # 🧠 Sentiment Analysis — BiLSTM Model
        Enter any English text (comment, tweet, review …) and the model
        will classify it as **Negative (0)**, **Neutral (1)**, or **Positive (2)**.
        """
    )
 
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Your Text",
                placeholder="Type or paste your comment here …",
                lines=4,
            )
            with gr.Row():
                submit_btn = gr.Button("Analyse Sentiment", variant="primary")
                clear_btn  = gr.ClearButton([text_input], value="Clear")
 
        with gr.Column(scale=1):
            label_output = gr.Label(label="Predicted Sentiment")
            probs_output = gr.Label(label="Confidence Scores")
 
    gr.Examples(
        examples=[
            ["This product is absolutely fantastic! I love it."],
            ["The delivery was okay, nothing special really."],
            ["Terrible experience. Completely broken and useless!"],
            ["It was alright. Not great but not bad either."],
            ["Best purchase I've made this year 😊"],
        ],
        inputs=text_input,
        label="Try an example →",
    )
 
    submit_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[label_output, probs_output],
    )
 
    gr.Markdown(
        """
        ---
        **Model:** Bidirectional LSTM (BiLSTM) &nbsp;|&nbsp;
        **Classes:** 0 = Negative &nbsp;·&nbsp; 1 = Neutral &nbsp;·&nbsp; 2 = Positive
        """
    )
 
# 6. Launch 
if __name__ == "__main__":
    demo.launch(
        share=False,       # set True for a public Gradio link (useful in Colab)
        server_port=7860,
    )