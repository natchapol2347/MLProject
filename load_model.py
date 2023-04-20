import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def get_paths_and_gts(partition_split_file):
    """
    read a string like(which can be found in each line of words.txt file):
    'a01-000u-00-00 ok 154 408 768 27 51 AT A'
    and extract 'a01-000u-00-00': location of the image with sub-folders, to read it from the directories
                'ok'            : processing status. ok means good, presumably.
                'A'             : ground truth text

    Then, pre-process using function defined above
    """
    # a list to store paths to images and ground truth texts
    paths_and_gts = []
    
    # open the file
    with open(partition_split_file) as f:
        # go through each line
        for line in f:
            # if a line is empty or commented with #, ignore that line
            if not line or line.startswith('#'):
                continue
            
            # in the text file, each line is seperated with '\n', so `strip` first
            # then string like 'a01-000u-00-00 ok 154 408 768 27 51 AT A' has to split to a list by spaces
            line_split = line.strip().split(' ')
            
            # the first item of the list contains path information, so split that by '-'
            directory_split = line_split[0].split('-')
            data_location ='words'
            # now use all the above and concatenate to a string to make a path to an image
            image_location = f'{data_location}/{directory_split[0]}/{directory_split[0]}-{directory_split[1]}/{line_split[0]}.png'
            
            # in a string like 'a01-000u-00-00 ok 154 408 768 27 51 AT A', text from 9th split is the ground truth text.
            gt_text = ' '.join(line_split[8:])
            
            # ignore a sample(image and ground truth text), if the ground truth has more than 16 letters
            # if len(gt_text) > 16:
                # continue
            
            # now, append the image location and ground truth text of that image as a list to 
            paths_and_gts.append([image_location, gt_text])
    
    return paths_and_gts

def add_padding(img, old_w, old_h, new_w, new_h):
    # print(img.shape)
    h1, h2 = int((new_h - old_h) / 2), int((new_h - old_h) / 2) + old_h
    w1, w2 = int((new_w - old_w) / 2), int((new_w - old_w) / 2) + old_w
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2, :] = img
    return img_pad


def fix_size(img, target_w, target_h):
        h, w = img.shape[:2]
        if w < target_w and h < target_h:
            img = add_padding(img, w, h, target_w, target_h)
        elif w >= target_w and h < target_h:
            new_w = target_w
            new_h = int(h * new_w / w)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        elif w < target_w and h >= target_h:
            new_h = target_h
            new_w = int(w * new_h / h)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        else:
            """w>=target_w and h>=target_h """
            ratio = max(w / target_w, h / target_h)
            new_w = max(min(target_w, int(w / ratio)), 1)
            new_h = max(min(target_h, int(h / ratio)), 1)
            new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img = add_padding(new_img, new_w, new_h, target_w, target_h)
        return img

def preprocess2(path, img_w, img_h):
    """ Pre-processing image for predicting """
    # try:
    img = cv2.imdecode(path, cv2.IMREAD_COLOR)
    print(img.shape)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img /= 255
# except:
#     pass
    return img
def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    # try:
    img = cv2.imread(path)
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    img /= 255
# except:
#     pass
    return img

letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_classes = len(letters) + 1
# print(num_classes)
def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))
from keras import layers
from keras import Model
from keras import backend as tf_keras_backend

tf_keras_backend.set_image_data_format('channels_last')
tf_keras_backend.image_data_format()
     
input_data = layers.Input(name='the_input', shape=(128,64,1), dtype='float32')  # (None, 128, 64, 1)

# Convolution layer (VGG)
iam_layers = layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max1')(iam_layers)  # (None,64, 32, 64)

iam_layers = layers.Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(2, 2), name='max2')(iam_layers)

iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max3')(iam_layers)  # (None, 32, 8, 256)

iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.Conv2D(512, (3, 3), padding='same', name='conv6')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)
iam_layers = layers.MaxPooling2D(pool_size=(1, 2), name='max4')(iam_layers)

iam_layers = layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(iam_layers)
iam_layers = layers.BatchNormalization()(iam_layers)
iam_layers = layers.Activation('relu')(iam_layers)

# CNN to RNN
iam_layers = layers.Reshape(target_shape=((32, 2048)), name='reshape')(iam_layers)
iam_layers = layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(iam_layers)

# RNN layer
gru_1 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(iam_layers)
gru_1b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(iam_layers)
reversed_gru_1b = layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (gru_1b)

gru1_merged = layers.add([gru_1, reversed_gru_1b])
gru1_merged = layers.BatchNormalization()(gru1_merged)

gru_2 = layers.GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = layers.GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
reversed_gru_2b= layers.Lambda(lambda inputTensor: tf_keras_backend.reverse(inputTensor, axes=1)) (gru_2b)

gru2_merged = layers.concatenate([gru_2, reversed_gru_2b])
gru2_merged = layers.BatchNormalization()(gru2_merged)

# transforms RNN output to character activations:
iam_layers = layers.Dense(80, kernel_initializer='he_normal', name='dense2')(gru2_merged)
iam_outputs = layers.Activation('softmax', name='softmax')(iam_layers)
def load_model():
    iam_model_pred = Model(inputs=input_data, outputs=iam_outputs)
    # iam_model_pred.summary()
    iam_model_pred.load_weights(filepath='my_handwriting_model.h5')
    return iam_model_pred
def processed_image_from_pic(test_image_path):
    test_images_processed = []
    # original_test_texts = []
    temp_processed_image = preprocess(path=test_image_path, img_w=128, img_h=64)
    test_images_processed.append(temp_processed_image.T)
    # original_test_texts.append(original_test_text)
    test_images_processed = np.array(test_images_processed)
    test_images_processed = test_images_processed.reshape(test_images_processed.shape[0], test_images_processed.shape[1], test_images_processed.shape[2], 1)
    return test_images_processed
def processed_image_from_data(test_image_path):
    test_images_processed = []
    # original_test_texts = []
    temp_processed_image = preprocess2(path=test_image_path, img_w=128, img_h=64)
    # cv2.imshow('Image', temp_processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_images_processed.append(temp_processed_image.T)
    # original_test_texts.append(original_test_text)
    test_images_processed = np.array(test_images_processed)
    test_images_processed = test_images_processed.reshape(test_images_processed.shape[0], test_images_processed.shape[1], test_images_processed.shape[2], 1)
    return test_images_processed
def numbered_array_to_text(numbered_array):
    numbered_array = numbered_array[numbered_array != -1]
    return "".join(letters[i] for i in numbered_array)
def pred(iam_model_pred,test_images_processed):
    test_predictions_encoded = iam_model_pred.predict(x=test_images_processed)
    test_predictions_decoded = tf_keras_backend.get_value(tf_keras_backend.ctc_decode(test_predictions_encoded,
                                                                                  input_length = np.ones(test_predictions_encoded.shape[0])*test_predictions_encoded.shape[1],
                                                                                  greedy=True)[0][0])
    return numbered_array_to_text(test_predictions_decoded[0])