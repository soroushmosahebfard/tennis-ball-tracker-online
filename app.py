from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Blur → HSV → Threshold
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower   = np.array([29, 100, 100])
        upper   = np.array([64, 255, 255])
        mask    = cv2.inRange(hsv, lower, upper)
        mask    = cv2.erode(mask, None, iterations=2)
        mask    = cv2.dilate(mask, None, iterations=2)

        # Contours → Circle around largest
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(frame, (int(x),int(y)), int(r), (0,255,0), 2)
                cv2.circle(frame, (int(x),int(y)), 5, (0,0,255), -1)

        # Shi–Tomasi corners
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            for pt in np.int0(corners):
                x, y = pt.ravel()
                cv2.circle(frame, (x,y), 4, (255,255,0), 2)

        # JPEG-encode & stream
        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
