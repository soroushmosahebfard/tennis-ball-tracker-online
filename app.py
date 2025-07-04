from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import base64
import io
import os

app = Flask(__name__)

# Detection pipeline: color + Hough + Shi-Tomasi

def detect_ball_and_corners(frame):
    proc = frame.copy()

    # 1) HSV color threshold for yellow/chartreuse tennis ball
    blurred = cv2.GaussianBlur(proc, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 2) Contour-based detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r > 10:
            cv2.circle(proc, (int(x), int(y)), int(r), (0, 255, 255), 2)
            detected = True

    # 3) Fallback: Hough Circle if color detection failed
    if not detected:
        gray2 = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.medianBlur(gray2, 5)
        circles = cv2.HoughCircles(
            gray2,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=200
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            # draw only the first detected circle
            x, y, r = circles[0]
            cv2.circle(proc, (x, y), r, (0, 255, 255), 2)

    # 4) Shi-Tomasi corner detection
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        for pt in corners.reshape(-1, 2):
            cv2.circle(proc, tuple(pt.astype(int)), 3, (255, 0, 0), -1)

    return proc

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # receive image dataURL
    data_url = request.json.get('image', '')
    _, b64data = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64data)

    # decode to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # run detection
    processed = detect_ball_and_corners(frame)

    # stack original + processed
    stacked = np.hstack((frame, processed))

    # JPEG-encode and send back
    ret, buf = cv2.imencode('.jpg', stacked)
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
