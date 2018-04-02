# USAGE
# python test_network.py --model /content/code/sketch_classification.model --image apple_test.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#names = ['airplane', 'alarm clock', 'angel', 'ant', 'apple']
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
# load the image
test_image = cv2.imread(args["image"])
orig = test_image.copy()

# pre-process the image for classification
test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(64,64))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255

test_image= np.expand_dims(test_image, axis=3) 
test_image= np.expand_dims(test_image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
index = model.predict_classes(test_image)
result = names[index[0]]
prob = model.predict(test_image)[0][index]
#print(model.predict(test_image))
print(result, prob)
# build the label
label = "{}: {:.2f}%".format(result, prob[0] * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (150, 300),  cv2.FONT_HERSHEY_SIMPLEX,
	1.0, (255, 0, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
