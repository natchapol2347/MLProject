import io
import base64
import tensorflow as tf
import numpy as np
import cv2
# from cv2 import cv
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import load_model as lm
from urllib.request import urlopen
import gpttest as gpt

model = lm.load_model()
app = Flask(__name__)
CORS(app)

@app.route("/api/recognize", methods=["POST"])
def recognize():
    global model
    # Get the image from the request
    test_id = request.values.get('image')
    byteArray = bytearray(base64.b64decode(test_id))
    picbyte = bytes(byteArray)
    img_array = np.frombuffer(picbyte, np.uint8)
   
    image = lm.processed_image_from_data(img_array)
    pred = lm.pred(model,image)
    print(pred)
    
    predicted_word= pred
    poem = gpt.get_reply("make me a poem with this word as a keyword: "+predicted_word)
    print("word: {predicted_word}, poem:{poem}".format(predicted_word = predicted_word, poem = poem))
    return {"word": predicted_word, "poem":poem}

if __name__ == "__main__":
    app.run(debug=True)
