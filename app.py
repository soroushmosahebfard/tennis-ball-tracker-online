from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open video device')


def detect_ball_and_corners(frame):
    """
    Returns a processed copy of the frame with
    a yellow circle around the ball and corner points.
    """
    proc = frame.copy()
    # Tennis-ball detection (yellow range)
    blurred = cv2.GaussianBlur(proc, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            cv2.circle(proc, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # Corner detection (Shi-Tomasi)
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        for pt in corners:
            x1, y1 = pt.ravel()
            cv2.circle(proc, (int(x1), int(y1)), 3, (255, 0, 0), -1)

    return proc


def gen_frames():
    """Video streaming generator: yields combined original + processed frames."""
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Detect ball and corners on a copy
        processed = detect_ball_and_corners(frame)

        # Stack original and processed side by side
        combined = np.hstack((frame, processed))

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
