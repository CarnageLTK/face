from pathlib import Path

import cv2, face_recognition, pickle, threading, time

END = False

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("know_faces").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


name = input("persons name: ")

Path.mkdir(Path("know_faces/"+name))

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
        
def capture():
    count = 0
    last = time.time()
    while END == False:
        if time.time() - last > 0.5:
            out = cv2.imwrite("know_faces/"+name+'/capture'+str(count)+'.jpg', frame)
            last = time.time()
            count += 1

thread = threading.Thread(target=capture)
thread.start()

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        vc.release()
        cv2.destroyWindow("preview")
        END = True
        break

encode_known_faces()