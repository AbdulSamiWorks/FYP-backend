from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client, handle_file
import tempfile, os, base64
from dotenv import load_dotenv
import openai

# Load secrets from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Hugging Face Space
client = Client("aitoolsami/FYP")

# Classify image using OpenAI GPT-4o Vision
def classify_with_openai(image_path):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this a human eye's retina image? Respond with only 'retinal' or 'other'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
        )

        result = response.choices[0].message.content.strip().lower()
        if "retinal" in result:
            return {"type": "retinal", "confidence": 0.99}
        else:
            return {"type": "other", "confidence": 0.99}

    except Exception as e:
        return {"error": f"OpenAI Vision error: {str(e)}"}

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
        tmp_path = tmp.name

    try:
        # Step 1: Use OpenAI Vision
        result = classify_with_openai(tmp_path)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # 🛠️ Unified response structure for non-retinal images
        if result["type"] != "retinal":
            return jsonify({
                "type": "other",
                "confidence": result["confidence"],
                "prediction": {
                    "disease": "Other",
                    "confidence": result["confidence"],
                    "advice": "The uploaded image does not appear to be a retinal image. Please upload a valid fundus photo."
                }
            })

        # Step 2: Send to Hugging Face if retinal
        hf_result = client.predict(
            img=handle_file(tmp_path),
            api_name="/predict"
        )

        return jsonify({
            "type": "retinal",
            "confidence": result["confidence"],
            "prediction": hf_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(debug=True)
