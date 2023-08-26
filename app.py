import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
import joblib


vectorizer = joblib.load('count_vectorizer.pkl')

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    new_review = request.form.get('Review_text')
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = vectorizer.transform(new_corpus).toarray()
    new_y_pred = model.predict(new_X_test)
    return render_template("index.html", prediction_text = "Review Rated to {}".format(new_y_pred))

if __name__ == "__main__":
    flask_app.run(debug=True)



