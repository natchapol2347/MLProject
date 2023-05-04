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
    # print(picbyte)
    img_array = np.frombuffer(picbyte, np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # print(img.shape)
    # image = lm.processed_image_from_pic(r"C:\Users\USER\Pictures\Camera Roll\ohmha.png")
    # image = lm.processed_image_from_pic(r"C:\Users\USER\MLProject\words\l04\l04-174\l04-174-01-03.png")
    image = lm.processed_image_from_data(img_array)
    # print(image)
    pred = lm.pred(model,image)
    print(pred)
    # Open a file for writing
    # with open('file.txt', 'w') as f:

    #     # Write some text to the file
    #     f.write(picbyte)

    # Decode image from array using cv2.imdecode()
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Display image using cv2.imshow()
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # req = urlopen(test_id)
    # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    # img = cv2.imdecode(arr, -1) # 'Load it as it is'
    # cv2.imshow('lalala', img)
    # if cv2.waitKey() & 0xff == 27: quit()
    # print(request.json['image'])
    # image_data = np.array(request.json['image'], dtype=np.uint8)

    # # Decode the image and preprocess it
    # img = lm.processed_image_from_pic(image_data)
    # predicted_word = lm.pred(img)
    predicted_word= pred
    poem = gpt.get_reply("make me a poem with this word as a keyword: "+predicted_word)
    return {"word": predicted_word, "poem":poem}

if __name__ == "__main__":
    app.run(debug=True)
