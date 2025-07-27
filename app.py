# app.py
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import tempfile

app = Flask(__name__)

client = Client("aitoolsami/FYP")  # Your HF Space slug

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files["file"]

    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)

        # Positional argument (not named)
        result = client.predict(
            [handle_file(tmp.name)],
            api_name="/predict"
        )

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run()
