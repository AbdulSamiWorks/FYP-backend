from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client, handle_file
import tempfile
import os

app = Flask(__name__)
CORS(app)

client = Client("aitoolsami/FYP")  # Replace with your actual HuggingFace space if needed

@app.route("/")
def home():
    return "✅ Flask backend is running"

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        try:
            result = client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
