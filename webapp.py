import argparse
import io
import os
import datetime
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, redirect, copy_current_request_context
from threading import Thread

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()

    # Perform object detection on the frames using YOLOv5 model
    results = model(frames)

    # Draw bounding boxes on the objects detected in the frames
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        results.render([i])
        boxes = results.xyxy[i].numpy()

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{results.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frames[i] = frame

    # Save the processed frames as a new video
    now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
    output_path = f"static/{now_time}.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()

    return output_path


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        if file.mimetype.startswith('image'):
            # Process image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            results = model([img])

            results.render()  # updates results.imgs with boxes and labels
            now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
            img_savename = f"static/{now_time}.png"
            Image.fromarray(results.ims[0]).save(img_savename)
            return redirect(img_savename)

        elif file.mimetype.startswith('video'):
            # Process video
            video_path = f"static/{datetime.datetime.now().strftime(DATETIME_FORMAT)}.mp4"
            file.save(video_path)

            # Perform video processing in a separate thread
            @copy_current_request_context
            def process_video_in_thread():
                output_path = process_video(video_path)
                if output_path:
                    return redirect(output_path)

            thread = Thread(target=process_video_in_thread)
            thread.start()

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load YOLOv5 model
    model.eval()
    app.run(host="0.0.0.0", port=args.port)