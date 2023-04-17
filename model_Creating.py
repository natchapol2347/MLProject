import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
import tensorflow as tf
from keras import layers, models


def load_iam_dataset(data_path, img_height, img_width):
    images = []
    labels = []
    
    with open(os.path.join(data_path, 'words.txt')) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(' ')
            file_path = os.path.join(data_path, parts[0] + '.png')
            
            if not os.path.exists(file_path):
                continue
            
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            images.append(img)
            labels.append(parts[-1])

    images = np.array(images).reshape(-1, img_height, img_width, 1).astype('float32') / 255
    return images, labels

# Set your IAM dataset path
iam_data_path = "/path/to/iam_dataset"

# Set your desired image height and width
img_height = 64
img_width = 128

# Load and preprocess the dataset
images, labels = load_iam_dataset(iam_data_path, img_height, img_width)

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

def create_model(input_shape, num_classes):
    input_data = layers.Input(shape=input_shape, name='input')
    
    # CNN layers
    cnn = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_data)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    cnn = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
    cnn = layers.MaxPooling2D((2, 2))(cnn)
    
    # Prepare output for RNN
    shape = cnn.get_shape().as_list()
    rnn_input = layers.Reshape(target_shape=(shape[1], shape[2] * shape[3]))(cnn)
    
    # RNN layers
    rnn = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.25))(rnn_input)
    rnn = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.25))(rnn)
    
    # Output layer
    output = layers.Dense(num_classes + 1, activation='softmax', name='output')(rnn)

    model = models.Model(inputs=input_data, outputs=output)
    return model


input_shape = (img_height, img_width, 1)
model = create_model(input_shape, num_classes)


def ctc_loss(y_true, y_pred):
    return tf.nn.ctc_loss(y_true, y_pred, label_length=None, logit_length=None, logits_time_major=False, unique=None, blank_index=-1, name=None)

def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss_value = ctc_loss(y_batch, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


epochs = 10
batch_size = 32
optimizer = Adam()

# Prepare the dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# Train the model
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value = train_step(x_batch, y_batch)
        print(f"Step {step + 1}: loss = {loss_value.numpy()}")

import numpy as np

def ctc_decode(y_pred):
    decoded_labels = []
    for i in range(y_pred.shape[0]):
        decoded, _ = tf.nn.ctc_greedy_decoder(y_pred[i:i+1].transpose((1, 0, 2)), [y_pred.shape[1]])
        decoded_labels.append([x[0] for x in decoded[0].numpy()])
    return np.asarray(decoded_labels)

y_pred = model.predict(x_test)
decoded_labels = ctc_decode(y_pred)
predicted_words = label_encoder.inverse_transform(decoded_labels)

correct_predictions = np.sum(y_test == predicted_words)
total_predictions = len(y_test)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

num_samples = 10

for i in range(num_samples):
    plt.imshow(x_test[i].reshape(img_height, img_width), cmap="gray")
    plt.title(f"True: {y_test[i]}\nPredicted: {predicted_words[i]}")
    plt.axis("off")
    plt.show()


model.save("my_handwriting_model.h5")
