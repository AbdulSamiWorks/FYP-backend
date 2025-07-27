# app.py
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile

app = Flask(__name__)

# Hugging Face Space slug (confirm this matches your Space)
client = Client("aitoolsami/FYP")

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files["file"]

    try:
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)

            # Predict using Hugging Face Space (expects list as input)
            result = client.predict(
                [handle_file(tmp.name)],  # Wrap in list for positional args
                api_name="/predict"
            )

        return jsonify({"prediction": result})

    except Exception as e:
        # Return error details for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
