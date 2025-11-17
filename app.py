from flask import Flask, render_template, request, redirect, url_for
import whisper
import re

app = Flask(__name__)

# Load Whisper model once (medium for balance)
model = whisper.load_model("base")

def extract_action_items_and_decisions(transcript):
    """
    Simple extraction rules.
    You can replace with NLP models later.
    """
    sentences = re.split(r'[.?!]', transcript)
    action_items = []
    decisions = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Action items keywords
        if any(word in s.lower() for word in ["will", "need to", "should", "arrange", "send", "prepare", "schedule"]):
            action_items.append(s)
        # Decision keywords
        if any(word in s.lower() for word in ["decided", "approved", "confirmed", "agreed", "extended", "rejected"]):
            decisions.append(s)

    if not action_items:
        action_items.append("No clear action items found.")
    if not decisions:
        decisions.append("No clear decisions found.")

    return action_items, decisions

@app.route("/", methods=["GET", "POST"])
def index():
    transcript, action_items, decisions = None, [], []
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            # Save temp file
            filepath = "uploaded_audio.wav"
            file.save(filepath)

            # Transcribe
            result = model.transcribe(filepath)
            transcript = result["text"]

            # Extract insights
            action_items, decisions = extract_action_items_and_decisions(transcript)

    return render_template("index.html",
                           transcript=transcript,
                           action_items=action_items,
                           decisions=decisions)

if __name__ == "__main__":
    app.run(debug=True)
