from flask import Flask, send_from_directory, make_response
import os

app = Flask(__name__, static_folder='')

@app.after_request
def add_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    if 'Content-Security-Policy' in resp.headers:
        resp.headers.pop('Content-Security-Policy', None)
    return resp

@app.route('/<path:p>')
def serve_files(p):
    return send_from_directory('.', p)

if __name__ == '__main__':
    print("🌍 Serving map at http://127.0.0.1:8000/")
    os.system("start http://127.0.0.1:8000/data/outputs/map_shapes_lstm.html")
    app.run(port=8000, debug=False)
