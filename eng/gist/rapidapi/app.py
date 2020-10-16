"""
https://towardsdatascience.com/develop-and-sell-a-python-api-from-start-to-end-tutorial-9a038e433966
"""
import json
import os
import pandas as pd

from flask import Flask, request, send_from_directory, abort

UPLOAD_DIRECTORY = "."
CONVERSIONS = ["h5", "pkl", "feather", "parquet"]

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "This app provides data format transformation!"


@app.route("/download", methods=["POST"])
def download_file():
    provided_data = request.args.get("file")
    if provided_data is None:
        return "Please enter valid excel format ", 400
    provided_format = request.form.get("format")
    if provided_format is None:
        return f"Please enter valid format to convert.", 400
    if provided_format not in CONVERSIONS:
        return f"Please enter valid format to convert. Can be {list(CONVERSIONS)}", 400

    df = pd.read_csv(provided_data)
    file_name = f"converted.{provided_format}"
    path = f"./{file_name}"

    if provided_format == "h5":
        df.to_hdf(path, file_name, mode="w")
    elif provided_format == "pkl":
        df.to_pickle(path)
    elif provided_format == "parquet":
        df.to_pickle(path)
    elif provided_format == "feather":
        df.to_pickle(path)

    try:
        return send_from_directory(UPLOAD_DIRECTORY, filename=file_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)
    finally:
        remove_file(path)


def remove_file(new_file):
    path = os.path.join(UPLOAD_DIRECTORY, new_file)
    os.remove(path)


@app.route("/upload", methods=["POST"])
def upload_file():
    filepath = request.args.get("file")
    print("filename: %s" % filepath)
    # if provided_data is None:
    #    return "Please enter valid file name", 400

    print("File location: %s" % os.path.dirname(os.path.abspath(__file__)))
    print("Working directory: %s" % os.path.abspath(os.getcwd()))

    data = pd.read_csv(filepath)
    transformed = data.to_json()
    result = {"result": transformed}
    return json.dumps(result)


if __name__ == "__main__":
    app.run()
