import io
import base64
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import load_model as lm

model = lm.load_model()
app = Flask(__name__)
CORS(app)

@app.route("/api/recognize", methods=["POST"])
def recognize():
    # Get the image from the request
    # print(request.json['image'])
    # image_data = np.array(request.json['image'], dtype=np.uint8)

    # # Decode the image and preprocess it
    # img = lm.processed_image_from_pic(image_data)
    # predicted_word = lm.pred(img)
    predicted_word= "hey"

    return jsonify({"word": predicted_word})

if __name__ == "__main__":
    app.run(debug=True)
