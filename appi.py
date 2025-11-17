from flask import Flask, render_template, request
import os
import whisper
import spacy

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Whisper model (you can switch to "base" or "small" if system is slow)
whisper_model = whisper.load_model("tiny")
# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

def extract_action_items_and_decisions(text):
    """
    Simple NLP-based extraction:
    - Action items: look for verbs like 'do', 'prepare', 'send', etc.
    - Decisions: look for keywords like 'decided', 'approved', 'confirmed'
    """
    doc = nlp(text)
    action_items, decisions = [], []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        lower = sent_text.lower()
        if any(word in lower for word in ["do", "prepare", "send", "finish", "schedule", "complete"]):
            action_items.append(sent_text)
        if any(word in lower for word in ["decided", "approved", "confirmed", "finalized", "agreed"]):
            decisions.append(sent_text)

    return action_items, decisions


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Transcribe audio with Whisper
        result = whisper_model.transcribe(filepath)
        transcript = result["text"]

        # Extract action items & decisions
        action_items, decisions = extract_action_items_and_decisions(transcript)

        return render_template(
            "index.html",
            transcript=transcript,
            action_items=action_items,
            decisions=decisions,
            filename=file.filename,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
