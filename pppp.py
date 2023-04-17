import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
