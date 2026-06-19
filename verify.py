"""
verify.py
"""

import argparse
import os
import sys
import cv2
import numpy as np

KNOWN_DIR = "known_faces"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# LBPH "confidence" is actually a DISTANCE — lower is better.
# Empirically, < 70 is a good match, > 100 is almost certainly wrong.
THRESHOLD = 70.0


def load_known_faces():
    """Read every image in KNOWN_DIR and prepare training data."""
    if not os.path.isdir(KNOWN_DIR):
        print(f"[ERROR] '{KNOWN_DIR}/' not found. Run register.py first.")
        sys.exit(1)

    detector = cv2.CascadeClassifier(CASCADE)
    images, labels, label_map = [], [], {}
    next_id = 0

    for fname in sorted(os.listdir(KNOWN_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(KNOWN_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # If the stored image is already a tight crop, use it as-is;
        # otherwise try to detect the face again.
        faces = detector.detectMultiScale(img, 1.1, 5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            img = img[y:y + h, x:x + w]
        img = cv2.resize(img, (200, 200))

        if name not in label_map:
            label_map[name] = next_id
            next_id += 1
        images.append(img)
        labels.append(label_map[name])

    if not images:
        print(f"[ERROR] No registered faces found in '{KNOWN_DIR}/'.")
        sys.exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    inv_map = {v: k for k, v in label_map.items()}
    print(f"[INFO] Loaded {len(images)} reference image(s): "
          f"{list(label_map.keys())}")
    return recognizer, inv_map


def annotate(frame, x, y, w, h, text, ok):
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - 28), (x + w, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def verify_frame(frame, recognizer, inv_map, detector):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    results = []
    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        label, distance = recognizer.predict(roi)
        if distance < THRESHOLD:
            name = inv_map.get(label, "?")
            text = f"VERIFIED: {name} ({distance:.1f})"
            ok = True
        else:
            text = f"UNVERIFIED ({distance:.1f})"
            ok = False
        annotate(frame, x, y, w, h, text, ok)
        results.append((text, ok, distance))
    return results


def run_webcam():
    recognizer, inv_map = load_known_faces()
    detector = cv2.CascadeClassifier(CASCADE)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)
    print("Press ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        verify_frame(frame, recognizer, inv_map, detector)
        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def run_image(path):
    recognizer, inv_map = load_known_faces()
    detector = cv2.CascadeClassifier(CASCADE)
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read {path}")
        sys.exit(1)
    results = verify_frame(frame, recognizer, inv_map, detector)
    if not results:
        print("[RESULT] No face detected -> UNVERIFIED")
    else:
        for text, ok, dist in results:
            print(f"[RESULT] {text}  (status={'OK' if ok else 'FAIL'})")
    out = "verify_output.jpg"
    cv2.imwrite(out, frame)
    print(f"[INFO] Annotated image saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Run one-shot verification on this file")
    args = p.parse_args()
    if args.image:
        run_image(args.image)
    else:
        run_webcam()
