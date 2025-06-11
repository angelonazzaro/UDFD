import requests

from datetime import datetime

from flask import Flask, render_template, request, jsonify

app = Flask("UDFD")


def log(message: str) -> None:
    today = datetime.today()
    with open(f"logs/app_{today.strftime('%Y-%m-%d')}.txt", "a") as f:
        f.write(f"{today.strftime('%H:%M:%S')} | {message}\n")


@app.route("/")
def hello() -> str:
    log(f"User accessed {request.path} with method {request.method}")
    return render_template("index.html")


# Proxy for the inference service
@app.route("/api/classify", methods=["POST"])
def classify():
    log(f"User accessed {request.path} with method {request.method}")
    files = request.files
    data = {"upload": request.form.get("upload", "false")}
    response = requests.post(
        "http://inference:5555/classify",
        files=files,
        data=data,
    )
    return jsonify(response.json())


@app.errorhandler(404)
def page_not_found(error) -> (str, int):
    log(f"User tried to access a non-existing page: '{request.path}'")
    return (
        render_template(
            "error.html",
            error="Page Not Found",
            description=f'The URL "{request.path}" does not exist.',
        ),
        404,
    )


@app.errorhandler(405)
def page_not_found(error) -> (str, int):
    log(
        f"User tried to access '{request.path}' with an unsupported method:"
        f" '{request.method}'"
    )
    return (
        render_template(
            "error.html",
            error="Method Not Allowed",
            description="The method is not allowed for the requested URL.",
        ),
        405,
    )


def main() -> None:
    app.run(host="0.0.0.0", port=5500, debug=True)


if __name__ == "__main__":
    log("Server started")
    main()
