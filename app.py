from flask import Flask, render_template, request, send_file
import cv2, numpy as np
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # receive base64-style data URL
    data_url = request.json['image']
    header, b64data = data_url.split(',', 1)
    img_data = np.frombuffer(
        base64.b64decode(b64data), dtype=np.uint8
    )
    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    # --- your tennis-ball detection logic ---
    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, (29,86,6), (64,255,255))
    mask    = cv2.erode(mask, None, iterations=2)
    mask    = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x,y),r) = cv2.minEnclosingCircle(c)
        if r > 10:
            cv2.circle(frame, (int(x),int(y)), int(r), (0,255,255), 2)

    # encode back to JPEG
    ret, buf = cv2.imencode('.jpg', frame)
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
