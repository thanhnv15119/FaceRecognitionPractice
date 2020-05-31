import face_recognition
import pickle
import cv2
import sys


cap= cv2.VideoCapture(0)

encodings_path="encodings.pickle"
image="data-example/00001.png"
detection_method="haarcascade_frontalface_default.xml"

print("[INFO] loading encodings...")
data = pickle.loads(open(encodings_path,"rb").read())

while True:
    ret, image = cap.read()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] recognzing faces..")
    boxes = face_recognition.face_locations(rgb, model=detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names= []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:

            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            counts={}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
                name = max(counts, key= counts.get)

        names.append(name)

    for ((top,right,bottom, left), name) in zip(boxes,names):
        cv2.rectangle(image, (left,top), (right, bottom), (0, 225, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,225, 0), 2)

    cv2.imshow("RESULT", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    







