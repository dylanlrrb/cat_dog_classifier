from flask import Flask, redirect, url_for, jsonify
from waitress import serve
import sys

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('static', filename='index.html'))

@app.route("/classify")
def main():
  # functions should use model in the location it is written to in the model folder for consistency
    return jsonify(
        mapping={"1": "Cat 🐱", "2": "Dog 🐶"},
        ranking=["2", "1"],
        certainty=["67.9", "13.5"],
        speed="823"
    )

def main():
  if len(sys.argv) >= 2 and sys.argv[1] == 'ssl':
    app.run("0.0.0.0", port=8080, debug=True, ssl_context='adhoc')
  serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
  main()