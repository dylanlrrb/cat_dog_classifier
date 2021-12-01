from flask import Flask, redirect, url_for, jsonify, request
from waitress import serve
from classify import classify
import sys
from PIL import Image
import base64, io
import re
import random
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('static', filename='index.html'))

@app.route("/classify", methods=['POST'])
def classify_image():
  # write image to disk
  hash = random.getrandbits(128)

  data_url = request.data.decode('utf-8')
  base64_data = re.sub('^data:image/.+;base64,', '', data_url)
  byte_data = base64.b64decode(base64_data)
  image_data = io.BytesIO(byte_data)
  img = Image.open(image_data)
  img = img.convert('RGB')
  img.save(f'./backend/{hash}.jpg', 'JPEG')
  
  data = classify(f'./backend/{hash}.jpg')
  # remove image from disk
  os.remove(f'./backend/{hash}.jpg')

  return jsonify(data)


def main():
  if len(sys.argv) > 1 and sys.argv[1] == 'ssl':
    app.run("0.0.0.0", port=8080, debug=True, ssl_context='adhoc')
  serve(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
  main()