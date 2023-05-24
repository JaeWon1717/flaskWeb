"""
Simple app to upload an image or video via a web form 
and view the inference results on the media in the browser.
"""
import argparse
import io
import os
import datetime
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


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

            # Perform face detection on the video
            cap = cv2.VideoCapture(video_path)
            faces = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform face detection on the frame (e.g., using Haar Cascade, Dlib, MTCNN, etc.)
                # Append the detected faces to the `faces` list

                # Draw rectangles on the face regions
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Display the frame with rectangles
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            return redirect(video_path)

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat