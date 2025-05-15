from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import json
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("Lohit20/biobert-v1.2-base-cased-v1.2-ner")
model = AutoModelForTokenClassification.from_pretrained("Lohit20/biobert-v1.2-base-cased-v1.2-ner")

bio_tags = ['I-LF', 'B-AC', 'B-LF', 'O'] 

entity_type_map = {
    'B-LF': 'Begining of Long Form',
    'I-LF': 'Inside Long Form',
    'B-AC': 'Abbreviation/Acronym',
    'O': 'Other' 
}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    return get_chat_response(msg)

def get_chat_response(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, axis=2)

    predicted_labels = [bio_tags[p] for p in predictions[0].numpy()]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    for token, label in zip(tokens, predicted_labels):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        if token.startswith("##") and entities:
            entities[-1]["token"] += token[2:]
        else:
            entity_type = entity_type_map.get(label, 'Unknown') 
            entities.append({"token": token, "label": label, "entity_type": entity_type})
    
    response = """
    <table style='width:100%; border-collapse: collapse; margin: 10px 0; border-radius: 6px; overflow: hidden; font-size: 0.9em;'>
        <thead>
            <tr style='background-color: #333; color: white;'>
                <th style='padding: 12px; text-align: left; border-bottom: 1px solid #444;'>Token</th>
                <th style='padding: 12px; text-align: left; border-bottom: 1px solid #444;'>Label</th>
                <th style='padding: 12px; text-align: left; border-bottom: 1px solid #444;'>Entity Type</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for entity in entities:
        response += f"""
            <tr style='background-color: #383838; color: white;'>
                <td style='padding: 12px; text-align: left; border-bottom: 1px solid #444; word-break: break-word;'>{entity['token']}</td>
                <td style='padding: 12px; text-align: left; border-bottom: 1px solid #444; word-break: break-word;'>{entity['label']}</td>
                <td style='padding: 12px; text-align: left; border-bottom: 1px solid #444; word-break: break-word;'>{entity['entity_type']}</td>
            </tr>
        """
    
    response += "</tbody></table>"
    
    predictions_logging_file(entities)

    return response

def predictions_logging_file(entities_data, log_file="token_predictions.json"): 
    data = {
        "timestamp": datetime.now().isoformat(),
        "bio_ner_tags": entities_data
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(data) + "\n")

if __name__ == '__main__':
    app.run(debug=True)