import cv2
import dlib

def multiface_detector(img):
    # Load the cascade
    face_cascade = dlib.get_frontal_face_detector()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade(gray, 1)
    if len(faces)<1:
        print("No faces detected")
    if len(faces)>1:
        print(" ALERT : More than one face detected ",len(faces)," faces detected   ")
    if len(faces)==1:
        print("One face detected")
    
    return faces

def main():
    multiface_detector()

if __name__ == '__main__':
    main()