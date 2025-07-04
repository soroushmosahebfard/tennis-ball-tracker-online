from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import io, base64, os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Receive image from client as data URL
    data_url = request.json.get('image', '')
    _, b64data = data_url.split(',', 1)
    img_data = base64.b64decode(b64data)

    # Decode to OpenCV image
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------- Tennis-ball detection (yellow range) -------
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
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
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    # ------- Corner detection -------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        for pt in corners:
            x1, y1 = pt.ravel()
            cv2.circle(frame, (int(x1), int(y1)), 3, (255, 0, 0), -1)

    # Encode annotated frame back to JPEG
    ret, buf = cv2.imencode('.jpg', frame)
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
