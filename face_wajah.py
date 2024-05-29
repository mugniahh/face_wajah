import face_recognition as fr
import cv2 as cv
import numpy as np

gambarZayn = fr.load_image_file("Foto/zayn.jpeg")
zaynEncoding = fr.face_encodings(gambarZayn)[0]

gambarHarry = fr.load_image_file("Foto/harry.jpeg")
harryEncoding = fr.face_encodings(gambarHarry)[0]

known_face_encodings = [
    zaynEncoding,
    harryEncoding
]

known_face_names = [
    "Zayn",
    "Harry"
]

unknown_images = fr.load_image_file("Foto/oneD2.jpeg")

face_locations = fr.face_locations(unknown_images)
face_encodings = fr.face_encodings(unknown_images, face_locations)
gambar_cv = cv.cvtColor(unknown_images, cv.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_face_encodings, face_encoding)

    name ="Tdk diKenal"

    face_distances = fr.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    if name=="Tdk diKenal":
        cv.rectangle(gambar_cv, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(gambar_cv, (left, top), (right, bottom), (0, 0, 255), cv.FILLED)

    else:
        cv.rectangle(gambar_cv, (left, top), (right, bottom), (255, 0, 0), 2)
        cv.rectangle(gambar_cv, (left, top), (right, bottom), (255, 0, 0), cv.FILLED)

    font = cv.FONT_HERSHEY_COMPLEX_SMALL
    cv.putText(gambar_cv, name, (left, top - 8), font, 0.7, (255, 255, 255), 1)

cv.imshow('Hasil Pengenalan wajah', gambar_cv)
cv.waitKey(0)