from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route("/finmind_proxy", methods=["POST"])
def finmind_proxy():
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"Content-Type": "application/json"}
    data = request.json
    response = requests.post(url, headers=headers, json=data)
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(port=5000)
