##from imutils import paths
##import numpy as np
##import imutils
##import pickle
##import cv2
##import os
##
##dataset = "dataset"
##
##embeddingFile = "output/embeddings.pickle" #initial name for embedding file
##embeddingModel = "openface-nn4.small2.v1.t7" #initializing model for embedding Pytorch
##
###initialization of caffe model for face detection
##prototxt = "model/deploy.prototxt"
##model =  "model/res10_300x300_ssd_iter_140000.caffemodel"
##
###loading caffe model for face detection
###detecting face from Image via Caffe deep learning
##detector = cv2.dnn.readNetFromCaffe(prototxt, model)
##
###loading pytorch model file for extract facial embeddings
###extracting facial embeddings via deep learning feature extraction
##embedder = cv2.dnn.readNetFromTorch(embeddingModel)
##
###gettiing image paths
##imagePaths = list(paths.list_images(dataset))
##
###initialization
##knownEmbeddings = []
##knownNames = []
##total = 0
##conf = 0.5
##
###we start to read images one by one to apply face detection and embedding
##for (i, imagePath) in enumerate(imagePaths):
##	print("Processing image {}/{}".format(i + 1,len(imagePaths)))
##	name = imagePath.split(os.path.sep)[-2]
##	image = cv2.imread(imagePath)
##	image = imutils.resize(image, width=600)
##	(h, w) = image.shape[:2]
##	#converting image to blob for dnn face detection
##	imageBlob = cv2.dnn.blobFromImage(
##		cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
##
##	#setting input blob image
##	detector.setInput(imageBlob)
##	#prediction the face
##	detections = detector.forward()
##
##	if len(detections) > 0:
##		i = np.argmax(detections[0, 0, :, 2])
##		confidence = detections[0, 0, i, 2]
##
##		if confidence > conf:
##			#ROI range of interest
##			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
##			(startX, startY, endX, endY) = box.astype("int")
##			face = image[startY:endY, startX:endX]
##			(fH, fW) = face.shape[:2]
##
##			if fW < 20 or fH < 20:
##				continue
##			#image to blob for face
##			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
##			#facial features embedder input image face blob
##			embedder.setInput(faceBlob)
##			vec = embedder.forward()
##			knownNames.append(name)
##			knownEmbeddings.append(vec.flatten())
##			total += 1
##
##print("Embedding:{0} ".format(total))
##data = {"embeddings": knownEmbeddings, "names": knownNames}
##f = open(embeddingFile, "wb")
##f.write(pickle.dumps(data))
##f.close()
##print("Process Completed")




from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset = "dataset"
embeddingFile = "output/embeddings.pickle" # Initial name for embedding file
embeddingModel = "openface-nn4.small2.v1.t7" # Initializing model for embedding Pytorch

# Initialization of caffe model for face detection
prototxt = "model/deploy.prototxt"
model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

# Loading caffe model for face detection
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

# Loading pytorch model file for extract facial embeddings
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Getting image paths
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

# We start to read images one by one to apply face detection and embedding
for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Converting image to blob for dnn face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Setting input blob image
    detector.setInput(imageBlob)
    # Prediction the face
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            # ROI range of interest
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Image to blob for face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            # Facial features embedder input image face blob
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print("Embedding: {0} ".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(embeddingFile), exist_ok=True)

# Save the embeddings and names to a file
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))

print("Process Completed")

