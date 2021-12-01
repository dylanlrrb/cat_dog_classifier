from flask import Flask, redirect, url_for, json, jsonify
from waitress import serve
import sys
from classify import classify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('static', filename='index.html'))

@app.route("/classify")
def classify_image():
  # write image to disk
  
  data = classify('./backend/cat.108.jpg')
  print(data)
  # remove image from disk
  return jsonify(data)


def main():
  if len(sys.argv) >= 2 and sys.argv[1] == 'ssl':
    app.run("0.0.0.0", port=8080, debug=True, ssl_context='adhoc')
  serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
  main()