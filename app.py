import requests as req
from flask import Flask, render_template, request, jsonify

API_KEY = "sk-sam-colega2-Zm6hD1cF5A"
API_URL = "https://give-bell-republic-responses.trycloudflare.com"

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        r = req.get(f"{API_URL}/health", timeout=10)
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 502


@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    try:
        image = request.files.get("image")
        mask = request.files.get("mask")
        if not image or not mask:
            return jsonify({"error": {"message": "Falta imagem ou mascara"}}), 400

        r = req.post(
            f"{API_URL}/v1/inpaint",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={
                "image": (image.filename, image.read(), image.content_type),
                "mask": (mask.filename, mask.read(), mask.content_type),
            },
            data={
                "prompt": request.form.get("prompt", "red leather"),
                "negative_prompt": request.form.get("negative_prompt", "blurry, low quality, deformed"),
                "strength": request.form.get("strength", "0.75"),
                "guidance_scale": request.form.get("guidance_scale", "7.5"),
                "steps": request.form.get("steps", "30"),
                "seed": request.form.get("seed", "-1"),
            },
            timeout=180,
        )
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 502


if __name__ == "__main__":
    print(f"\n  API remota: {API_URL}")
    print(f"  Abre: http://localhost:5001\n")
    app.run(host="0.0.0.0", port=5001)
