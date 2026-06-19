"""
register.py
-----------
Step 1: Register a user by saving a reference face image.

Usage:
    python register.py --name alice --image path/to/photo.jpg
    python register.py --name alice --webcam        # capture from camera
"""
import argparse
import os
import sys
import cv2

KNOWN_DIR = "known_faces"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def detect_and_crop_face(image):
    """Detect the largest face in the image and return the cropped face (BGR)."""
    detector = cv2.CascadeClassifier(CASCADE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                      minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick the largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return image[y:y + h, x:x + w]


def register_from_file(name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    face = detect_and_crop_face(img)
    if face is None:
        print("[ERROR] No face detected in image.")
        sys.exit(1)
    os.makedirs(KNOWN_DIR, exist_ok=True)
    out = os.path.join(KNOWN_DIR, f"{name}.jpg")
    cv2.imwrite(out, face)
    print(f"[OK] Registered '{name}' -> {out}")


def register_from_webcam(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)
    print("Press SPACE to capture, ESC to cancel.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("Register - press SPACE", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:           # ESC
            break
        if key == 32:           # SPACE
            face = detect_and_crop_face(frame)
            if face is None:
                print("No face detected, try again.")
                continue
            os.makedirs(KNOWN_DIR, exist_ok=True)
            out = os.path.join(KNOWN_DIR, f"{name}.jpg")
            cv2.imwrite(out, face)
            print(f"[OK] Registered '{name}' -> {out}")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="User name / id")
    p.add_argument("--image", help="Path to reference image")
    p.add_argument("--webcam", action="store_true", help="Capture from webcam")
    args = p.parse_args()

    if args.webcam:
        register_from_webcam(args.name)
    elif args.image:
        register_from_file(args.name, args.image)
    else:
        print("Provide either --image PATH or --webcam")
        sys.exit(1)
