from glob import glob
from os import path

import cv2
import face_recognition
from numpy import argmin, array


class FaceRec:
    def __init__(self):
        self.known_face_encodings = list()
        self.known_face_names = list()

        # Gotta go fast
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path: str) -> None:
        images_path = glob(path.join(images_path, "*.*"))
        print(f'Found {len(images_path)} images')

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = path.basename(img_path)
            (filename, ext) = path.splitext(basename)
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)

    def detect_known_faces(self, frame):
        face_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(face_frame_rgb)
        face_encodings = face_recognition.face_encodings(face_frame_rgb, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # figure out a face with the biggest possibility
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        face_locations = array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names
