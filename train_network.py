# USAGE
# python train_network.py --dataset "C:/Users/chandan/Desktop/Final_year_project/dataset/png" --model sketch_classification.model
# python train_network.py --dataset "/content/new_final_year_project/dataset/png" --model sketch_classification.model
# import the necessary packages
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from lenet import LeNet
from VGGNet import VGG
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
ROWS = 128
COLS = 128
CHANNELS = 1
EPOCHS = 25
INIT_LR = 1e-3
BS = 128
sketch_data_list = []

# initialize the data and labels
print("[INFO] loading images...")

# grab the image paths and randomly shuffle them
path = args["dataset"]
data_list = os.listdir(path)
#print("Categories:")
#print(data_list)
count = 0
for dataset in data_list:
	#print(dataset)
	dataset_path = os.path.join(path, dataset)
	#print("Individual category path:")
	#print(dataset_path)
	sketch_list = os.listdir(dataset_path)

	for sketch in sketch_list :
		#print("Sketch:")
		#print(sketch)
		count += 1
		if count%1000 == 0 :
		    #print("Count:")
		    print(count)  
		sketch_path = os.path.join(dataset_path, sketch)
		input_sketch = cv2.imread(sketch_path)
		input_sketch = cv2.cvtColor(input_sketch, cv2.COLOR_BGR2GRAY)
		input_sketch_rf = cv2.resize(input_sketch, (ROWS, COLS))
		sketch_data_list.append(input_sketch_rf)

sketch_data = np.array(sketch_data_list)
#print("\nFirst time:")
#print(sketch_data)
sketch_data = sketch_data.astype('float32')
sketch_data /= 255

#dimensional ordering
sketch_data = np.expand_dims(sketch_data, axis = 4)
print("\n Second time:")
print(sketch_data)

'''
sketch_data_normalized = preprocessing.normalize(sketch_data) #Scale/Normlaize#
print(sketch_data_normalized.shape)

# dimensional ordering
sketch_data_normalized = sketch_data_normalized.reshape(sketch_data.shape[0],img_rows, img_cols, num_channel)
print(sketch_data_normalized.shape)'''

#define classes 
num_of_classes = 2
num_of_samples = sketch_data.shape[0]
labels = np.ones((num_of_samples,), dtype ='int64')
#print("Labels : ", labels)
with open('Labels.csv', 'r') as f:
	reader = csv.reader(f, delimiter = ',')
	for row in reader:
		class_no = int(row[0])
		start = int(row[1])
		end = int(row[2])
		labels[start:end] = class_no
		#print(start,end)
#print("Labels : ", labels)

names = ['airplane', 'alarm clock']

'''
names = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm', 'armchair',
 'ashtray', 'axe', 'backpack', 'banana', 'barn', 'baseball bat', 'basket', 'bathtub',
 'bear (animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 
 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle opener', 'bowl', 'brain', 'bread',
 'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator',
 'camel', 'camera', 'candle', 'cannon', 'canoe', 'car (sedan)', 'carrot', 'castle', 'cat', 
 'cell phone', 'chair', 'chandelier', 'church', 'cigarette', 'cloud', 'comb', 'computer monitor', 
 'computer-mouse', 'couch', 'cow', 'crab', 'crane (machine)', 'crocodile', 'crown', 'cup', 'diamond', 
 'dog', 'dolphin', 'donut', 'door', 'door handle', 'dragon', 'duck', 'ear', 'elephant', 'envelope', 
 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fire hydrant', 'fish', 'flashlight', 'floor lamp', 
 'flower with stem', 'flying bird', 'flying saucer', 'foot', 'fork', 'frog', 'frying-pan', 'giraffe', 
 'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head', 'head-phones', 
 'hedgehog', 'helicopter', 'helmet', 'horse', 'hot air balloon', 'hot-dog', 'hourglass', 'house', 
 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder', 
 'laptop', 'leaf', 'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'megaphone', 
 'mermaid', 'microphone', 'microscope', 'monkey', 'moon', 'mosquito', 'motorbike', 'mouse (animal)', 
 'mouth', 'mug', 'mushroom', 'nose', 'octopus', 'owl', 'palm tree', 'panda', 'paper clip', 'parachute', 
 'parking meter', 'parrot', 'pear', 'pen', 'penguin', 'person sitting', 'person walking', 'piano', 
 'pickup truck', 'pig', 'pigeon', 'pineapple', 'pipe (for smoking)', 'pizza', 'potted plant', 
 'power outlet', 'present', 'pretzel', 'pumpkin', 'purse', 'rabbit', 'race car', 'radio', 
 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster', 'sailboat', 'santa claus', 
 'satellite', 'satellite dish', 'saxophone', 'scissors', 'scorpion', 'screwdriver', 
 'sea turtle', 'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skateboard', 
 'skull', 'skyscraper', 'snail', 'snake', 'snowboard', 'snowman', 'socks', 'space shuttle', 
 'speed-boat', 'spider', 'sponge bob', 'spoon', 'squirrel', 'standing bird', 'stapler', 
 'strawberry', 'streetlight', 'submarine', 'suitcase', 'sun', 'suv', 'swan', 'sword', 
 'syringe', 't-shirt', 'table', 'tablelamp', 'teacup', 'teapot', 'teddy-bear', 'telephone', 
 'tennis-racket', 'tent', 'tiger', 'tire', 'toilet', 'tomato', 'tooth', 'toothbrush', 'tractor', 
 'traffic light', 'train', 'tree', 'trombone', 'trousers', 'truck', 'trumpet', 'tv', 'umbrella', 
 'van', 'vase', 'violin', 'walkie talkie', 'wheel', 'wheelbarrow', 'windmill', 'wine-bottle', 
 'wineglass', 'wrist-watch', 'zebra']

names = ['airplane', 'alarm clock', 'angel', 'ant', 'apple', 'arm', 'armchair']
'''
# convert class labels to one-hot encoding
Y = np_utils.to_categorical(labels, num_of_classes)
print("Y : ", Y)

#Shuffle the dataset
x,y = shuffle(sketch_data,Y, random_state=2)
#print("x :", x)
#print("y : ", y)
# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print("X_train: ", X_train)
print("X_test: ", X_test)
print("Y_train: ", Y_train)
print("Y_test: ", Y_test)

# initialize the model
print("[INFO] compiling model...")
#model = LeNet.build(width=ROWS, height=COLS, depth=CHANNELS, classes=num_of_classes)
model = VGG.build(width=ROWS, height=COLS, depth=CHANNELS, classes=num_of_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss=categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

# train the network
print("[INFO] training network...")
hist = model.fit(X_train, Y_train, batch_size=BS, epochs=EPOCHS, verbose=1, validation_data=(X_test, Y_test))


# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"]) 
