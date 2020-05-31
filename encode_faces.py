from imutils import paths 
import face_recognition
import pickle
import cv2
import os

dataset = "dataset"
encodings = "encodings.pickle"
detection_method = "haarcascade_frontalface_default.xml"

print("[INFO] quantifying faces ...")
imagePaths = list(paths.list_images(dataset))

knownEncodings = []
knowNames = []

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] Processing Imag {}/{}".format(i+1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=detection_method)

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:

        knownEncodings.append(encoding)
        knowNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knowNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()




    
