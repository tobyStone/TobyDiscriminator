import gradio as gr
import torch
from train_discriminator import CharDiscriminator, preprocess_char
import spacy
from transformers import pipeline
import joblib
import numpy as np

# Configuration
CHAR_MODEL_PATH = 'model_dir/char_discriminator.pth'
SYNTAX_MODEL_PATH = 'syntax_discriminator.joblib'
HF_MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
MAX_LEN = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Char-level discriminator
char_vocab_size = 100  # Ensure this matches your trained model's vocab size
char_model = CharDiscriminator(vocab_size=char_vocab_size).to(device)
char_model.load_state_dict(torch.load(CHAR_MODEL_PATH, map_location=device))
char_model.eval()

# Load Syntax discriminator
syntax_model = joblib.load(SYNTAX_MODEL_PATH)
nlp = spacy.load('en_core_web_sm')

# Load Hugging Face pipeline
hf_pipeline = pipeline('text-classification', model=HF_MODEL_NAME)

# Ensemble prediction function
def ensemble_rank_text(text):
    try:
        # Character-level prediction
        char_input, _ = preprocess_char([text], MAX_LEN)
        char_prob = char_model(char_input.to(device)).item()

        # Syntax prediction
        doc = nlp(text)
        pos_counts = np.zeros(len(nlp.get_pipe("tagger").labels))
        for token in doc:
            pos_counts[token.pos] += 1
        syntax_prob = syntax_model.predict_proba([pos_counts])[0][1]

        # Transformer prediction
        transformer_result = hf_pipeline(text)[0]
        transformer_prob = transformer_result['score'] if transformer_result['label'] == 'LABEL_1' else 1 - transformer_result['score']

        # Ensemble weighting
        weights = {'char': 0.3, 'syntax': 0.3, 'transformer': 0.4}
        ensemble_score = (weights['char'] * char_prob +
                          weights['syntax'] * syntax_prob +
                          weights['transformer'] * transformer_prob)

        detailed_output = {
            'Ensemble Score': round(ensemble_score, 4),
            'Character-level': round(char_prob, 4),
            'Syntax-level': round(syntax_prob, 4),
            'Transformer-level': round(transformer_prob, 4)
        }

        return detailed_output

    except Exception as e:
        return {"Error": str(e)}

# Gradio Interface
iface = gr.Interface(
    fn=ensemble_rank_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text to evaluate...", label="Input Text"),
    outputs=gr.JSON(label="Detailed Ranking Scores"),
    title="StyleRank AI Ensemble Discriminator",
    description=(
        "Evaluate text similarity to Toby Stone's writing style using an "
        "ensemble discriminator (character-level, syntax, transformer-based). "
        "Scores closer to 1 indicate stronger stylistic similarity."
    )
)

iface.launch()
