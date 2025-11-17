import os
from flask import Flask, request, jsonify
from pydub import AudioSegment
import whisper
import tempfile
import re

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model (can use "base", "small", "medium", "large")
whisper_model = whisper.load_model("base")


# --------------------
# Utility Functions
# --------------------
def convert_to_wav(input_path, output_path):
    """Convert any audio to mono 16kHz WAV."""
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    sound.export(output_path, format="wav")


def validate_audio(file_path):
    """Check if audio file has valid duration."""
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) > 0
    except Exception:
        return False


def extract_action_items(text):
    """Extract action items from text."""
    items = re.findall(r"(?:Action:|TODO:|Task:)\s*(.*?)(?:\.|\n|$)", text, re.IGNORECASE)
    return [item.strip() for item in items if item.strip()]


def extract_key_decisions(text):
    """Extract key decisions from text."""
    decisions = re.findall(r"(?:Decision:|Decided:)\s*(.*?)(?:\.|\n|$)", text, re.IGNORECASE)
    return [d.strip() for d in decisions if d.strip()]


# --------------------
# Routes
# --------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save original file
    original_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(original_path)

    # Validate audio
    if not validate_audio(original_path):
        return jsonify({"error": "Invalid or empty audio file"}), 400

    # Convert to WAV for Whisper
    converted_path = os.path.join(UPLOAD_FOLDER, "converted.wav")
    convert_to_wav(original_path, converted_path)

    try:
        # Transcription
        result = whisper_model.transcribe(converted_path, word_timestamps=False)
        transcript = result.get("text", "").strip()

        # Extract structured data
        action_items = extract_action_items(transcript)
        key_decisions = extract_key_decisions(transcript)

        return jsonify({
            "status": "success",
            "transcript": transcript,
            "action_items": action_items,
            "key_decisions": key_decisions
        })

    except RuntimeError as e:
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    app.run(debug=True)
