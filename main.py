import cv2

from FaceRec import FaceRec

if __name__ == '__main__':
    face_rec = FaceRec()
    face_rec.load_encoding_images('resources/')

    # camera, if you have more cameras (I only have 1, increment the index)
    # alternatively can also use picture as input
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # keep detecting faces and figure out the names
        face_locations, face_names = face_rec.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            # draw a box
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Face recognition", frame)

        # await ESC key being pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
