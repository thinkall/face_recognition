import numpy as np
import face_recognition
from flask import Flask, request


def get_encoded_faces(img_numpy=None):
    # Get the face encodings for each face in each image file
    # Since there could be more than one face in each image, it returns a list of encodings.
    if img_numpy is not None:
        img_faces_encoding = face_recognition.face_encodings(img_numpy)
        return img_faces_encoding
 

def get_known_faces(img_files=[]):
    # Load the jpg file into numpy array
    known_faces_encoding = []
    for img_file in img_files:
        known_img = face_recognition.load_image_file(img_file)
        known_faces_encoding += get_encoded_faces(known_img)
    return known_faces_encoding


def compare_faces(known_faces_encoding, unknown_face_encoding):
    # Compare unkown face encoding in unknown image with faces encodings in known image
    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_faces_encoding, unknown_face_encoding)
    # return True if unknown face matched anyone in the known_faces array, False otherwise
    return True if True in results else False


# Initialize the Flask application
app = Flask(__name__)
app.known_faces = get_known_faces()


@app.route('/numpy', methods=['POST'])
def recognize_numpy():
    # recognize faces in numpy array
    img_numpy = np.array(request.json['img_numpy']).astype(np.uint8)
    unkown_faces_encodings = get_encoded_faces(img_numpy)
    results = []
    for unkown_face_encoding in unkown_faces_encodings:
        result = compare_faces(app.known_faces, unkown_face_encoding)
        if result:
            results.append(True)
        else:
            results.append(False)
    result = 1 if True in results else 0
    return {'result': result}


@app.route('/', methods=['GET'])
def index():
    return 'Welcome to face_recognition API!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
