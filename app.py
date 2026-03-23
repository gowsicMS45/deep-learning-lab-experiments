from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=False
)

EMOJI_MAP = {
    "joy":      "😊 Happy",
    "sadness":  "😢 Sad",
    "anger":    "😠 Angry",
    "fear":     "😨 Fear",
    "surprise": "😲 Surprise",
    "love":     "❤️ Love"
}

@app.route("/", methods=["GET", "POST"])
def home():
    emotion = ""
    confidence = 0
    if request.method == "POST":
        text = request.form["text"]
        result = emotion_classifier(text)[0]
        label = result["label"]
        confidence = round(result["score"] * 100, 2)
        emotion = EMOJI_MAP.get(label, label)
    return render_template("index.html", emotion=emotion, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)