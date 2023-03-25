import json

from flask import Flask, render_template

app = Flask(__name__)


with open("clusters.json") as f:
    DATUM = json.load(f)

@app.route("/team/<int:team_id>")
def team_analysis(team_id):
    team = DATUM[team_id]
    return render_template("team_analysis.html", team=team)


@app.route("/")
def index():
    return render_template("index.html", teams=DATUM)


if __name__ == "__main__":
    app.run(debug=True)
