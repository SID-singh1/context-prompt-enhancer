from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json()
    prompt = data.get("prompt", "")
    enhanced = prompt[::-1]  # placeholder reverse logic
    return jsonify({"original": prompt, "enhanced": enhanced})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
