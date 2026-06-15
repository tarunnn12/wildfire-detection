from flask import Flask, render_template, Response, jsonify
from detector import FireDetector
import cv2
import threading
import time
import datetime

app = Flask(__name__)

detector = FireDetector()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

state = {
    'detections': [],
    'frame_count': 0,
    'fire_count': 0,
    'smoke_count': 0,
    'last_alert': None,
    'status': 'monitoring',
    'fps': 0,
}
lock = threading.Lock()


def generate_frames():
    prev_time = time.time()
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, dets = detector.predict(frame)

        now = time.time()
        fps = 1 / (now - prev_time + 1e-9)
        prev_time = now

        with lock:
            state['detections'] = dets
            state['frame_count'] += 1
            state['fps'] = round(fps, 1)
            classes = [d['class'] for d in dets]
            if 'fire' in classes:
                state['fire_count'] += 1
                state['status'] = 'fire'
                state['last_alert'] = datetime.datetime.now().strftime('%H:%M:%S')
            elif 'smoke' in classes:
                state['smoke_count'] += 1
                state['status'] = 'smoke'
                state['last_alert'] = datetime.datetime.now().strftime('%H:%M:%S')
            else:
                state['status'] = 'clear'

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/state')
def get_state():
    with lock:
        return jsonify(state.copy())


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
