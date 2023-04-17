import io
import base64
import tensorflow as tf
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved model
model = load_model("my_handwriting_model.h5")

# Load the label encoder
# (assuming you saved it as a pickle file)
import pickle
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# CTC decoding function
def ctc_decode(y_pred):
    decoded_labels = []
    for i in range(y_pred.shape[0]):
        decoded, _ = tf.nn.ctc_greedy_decoder(y_pred[i:i+1].transpose((1, 0, 2)), [y_pred.shape[1]])
        decoded_labels.append([x[0] for x in decoded[0].numpy()])
    return np.asarray(decoded_labels)

@app.route("/api/recognize", methods=["POST"])
def recognize():
    # Get the image from the request
    image_data = request.files["image"].read()

    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the image and preprocess it
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 480))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Make a prediction using the model
    y_pred = model.predict(img)
    decoded_labels = ctc_decode(y_pred)

    # Convert the decoded label back to the original word
    predicted_word = label_encoder.inverse_transform(decoded_labels)[0]

    return jsonify({"word": predicted_word})

if __name__ == "__main__":
    app.run()
