from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# open default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open video device')

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        # blur + HSV + mask for green tennis ball
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = cv2.inRange(hsv, (29,86,6), (64,255,255))
        mask    = cv2.erode(mask, None, iterations=2)
        mask    = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x,y),r) = cv2.minEnclosingCircle(c)
            if r > 10:
                cv2.circle(frame, (int(x),int(y)), int(r), (0,255,255), 2)

        ret, buf = cv2.imencode('.jpg', frame)
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
