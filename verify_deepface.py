"""
verify_deepface.py  (OPTIONAL — more accurate)
"""

import argparse
import os
import sys
import cv2
import time

try:
    from deepface import DeepFace
except ImportError:
    print("Install with:  pip install deepface tf-keras")
    sys.exit(1)

KNOWN_DIR = "known_faces"
MODEL = "Facenet"          # other options: "ArcFace", "VGG-Face", "SFace"
DETECTOR = "ssd"           # ssd is fast AND reliable; opencv often misses faces
METRIC = "cosine"


def verify_pair(ref, probe):
    """Verify two face images. Returns result dict or None if face not found."""
    try:
        res = DeepFace.verify(img1_path=ref, img2_path=probe,
                              model_name=MODEL, detector_backend=DETECTOR,
                              distance_metric=METRIC, enforce_detection=True)
        return res  # dict with "verified": bool, "distance": float, "threshold": float
    except ValueError as e:
        # enforce_detection=True raises ValueError when no face is found
        print(f"  [WARN] Face not detected: {e}")
        return None


def run_pair(ref, probe):
    r = verify_pair(ref, probe)
    if r is None:
        print("[RESULT] FAILED — no face detected in one or both images.")
        return
    status = "VERIFIED" if r["verified"] else "UNVERIFIED"
    print(f"[RESULT] {status}  distance={r['distance']:.4f}  "
          f"threshold={r['threshold']:.4f}")


def preprocess_frame(frame):
    """Improve webcam frame quality for better face detection."""
    # Convert to LAB and apply CLAHE to lightness channel for better contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_webcam():
    refs = [os.path.join(KNOWN_DIR, f) for f in os.listdir(KNOWN_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not refs:
        print(f"No reference images in {KNOWN_DIR}/"); sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); sys.exit(1)

    print("Press SPACE to verify current frame, ESC to quit.")
    overlay_text = ""
    overlay_color = (255, 255, 255)
    overlay_until = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Show overlay result on frame for 3 seconds
        display = frame.copy()
        if time.time() < overlay_until and overlay_text:
            cv2.putText(display, overlay_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
            cv2.putText(display, overlay_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, overlay_color, 2)

        cv2.imshow("DeepFace - SPACE=verify ESC=quit", display)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 32:
            # Preprocess and save probe frame
            processed = preprocess_frame(frame)
            cv2.imwrite("_probe.jpg", processed)

            print("\n--- Verifying... ---")
            best_name, best_dist, verified = None, 1e9, False
            face_found = False

            for ref in refs:
                r = verify_pair(ref, "_probe.jpg")
                if r is not None:
                    face_found = True
                    if r["distance"] < best_dist:
                        best_dist = r["distance"]
                        best_name = os.path.splitext(os.path.basename(ref))[0]
                        verified = r["verified"]

            if not face_found:
                msg = "NO FACE DETECTED - try better lighting"
                print(f"[FAILED] {msg}")
                overlay_text = msg
                overlay_color = (0, 165, 255)  # orange
            elif verified:
                msg = f"VERIFIED: {best_name} (d={best_dist:.4f})"
                print(f"[VERIFIED] {best_name}  distance={best_dist:.4f}")
                overlay_text = msg
                overlay_color = (0, 255, 0)  # green
            else:
                msg = f"UNVERIFIED (d={best_dist:.4f})"
                print(f"[UNVERIFIED] closest={best_name} "
                      f"distance={best_dist:.4f}")
                overlay_text = msg
                overlay_color = (0, 0, 255)  # red

            overlay_until = time.time() + 3.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference")
    p.add_argument("--probe")
    p.add_argument("--webcam", action="store_true")
    a = p.parse_args()
    if a.webcam:
        run_webcam()
    elif a.reference and a.probe:
        run_pair(a.reference, a.probe)
    else:
        p.print_help()
