# app/app.py
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def home():
    # Path to generated map
    map_path = os.path.join("..", "data", "outputs", "map.html")
    if not os.path.exists(map_path):
        return "<h3>No map found. Please run visualize.py first to generate it.</h3>"
    
    # Read map HTML and embed inside a template
    with open(map_path, "r", encoding="utf-8") as f:
        map_html = f.read()

    return render_template("index.html", map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
