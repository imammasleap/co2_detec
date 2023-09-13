from flask import (
    Flask,
    render_template,
    redirect,
    request,
    send_from_directory,
    Response,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
import cv2
import yolov5
from yolov5.utils.plots import Annotator, colors


app = Flask(__name__)
app.config["UPLOAD_DIRECTORY"] = "static/uploads/"
app.config["THUMBNAILS_DIRECTORY"] = "static/thumbnails/"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 16MB
app.config["ALLOWED_EXTENSIONS"] = [".mp4"]

input_video_file_name = ""

model = yolov5.load("models/yolo/fire.pt")
if hasattr(model, "module"):
    names = model.module.names
else:
    names = model.names


# Create directory
try:
    os.makedirs(app.config["UPLOAD_DIRECTORY"])
    os.makedirs(app.config["THUMBNAILS_DIRECTORY"])
except OSError:
    print("Error on create directory")


@app.route("/")
def index():
    files = os.listdir(app.config["UPLOAD_DIRECTORY"])
    images = []
    for file in files:
        if os.path.splitext(file)[1].lower() in app.config["ALLOWED_EXTENSIONS"]:
            images.append(file)

    return render_template("index.html", images=images)


@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        if file:
            extension = os.path.splitext(file.filename)[1].lower()
            if extension not in app.config["ALLOWED_EXTENSIONS"]:
                return "Selected file is not a video."
            file.save(
                os.path.join(
                    app.config["UPLOAD_DIRECTORY"], secure_filename(file.filename)
                )
            )
    except RequestEntityTooLarge:
        return "File is larger than the 200MB limit."
    return redirect("/")


@app.route("/thumbnail/<filename>", methods=["GET"])
def thumbnail(filename):
    vcap = cv2.VideoCapture(
        os.path.join(app.config["UPLOAD_DIRECTORY"], secure_filename(filename))
    )
    res, im_ar = vcap.read()
    while res:
        res, im_ar = vcap.read()
        if im_ar.mean() > 60:
            res, im_ar = vcap.read()
            break

    cv2.imwrite(
        os.path.join(
            app.config["THUMBNAILS_DIRECTORY"],
            secure_filename(os.path.splitext(filename)[0] + ".jpg"),
        ),
        im_ar,
    )
    return send_from_directory(
        app.config["THUMBNAILS_DIRECTORY"], os.path.splitext(filename)[0] + ".jpg"
    )


@app.route("/object-detection/<filename>", methods=["GET"])
def object_detection(filename):
    global input_video_file_name
    input_video_file_name = filename
    return render_template(
        "video_player.html",
    )


def generate_smoke_detection_frames(file_name):
    cap = cv2.VideoCapture(file_name)
    model.conf = 0.2
    model.iou = 0.1

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: failed to capture image")
            break

        results = model(frame, augment=True)
        annotator = Annotator(frame, line_width=2, pil=not ascii)

        det = results.pred[0]
        if det is not None and len(det):
            confs = det[:, 4]
            clss = det[:, 5]

            combined_tuples = zip(det[:, :4], confs, clss)
            for xywh, conf, cls in combined_tuples:
                x, y, w, h = map(int, xywh)
                c = int(cls)

                if names[c] != "default":
                    label_name = "CO2" if names[c] == "smoke" else names[c]
                    label = f"{label_name} {conf:.2f}"
                    annotator.box_label(
                        (x, y, x + w, y + h), label, color=colors(c, True)
                    )

        im0 = annotator.result()
        image_bytes = cv2.imencode(".jpg", im0)[1].tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + image_bytes + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    filename = os.path.join(
        app.config["UPLOAD_DIRECTORY"], secure_filename(input_video_file_name)
    )
    method = generate_smoke_detection_frames(filename)

    return Response(method, mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
