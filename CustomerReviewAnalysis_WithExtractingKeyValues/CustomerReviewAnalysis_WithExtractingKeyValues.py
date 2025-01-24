import json
import logging
from flask import request, jsonify, Flask,Request
from textblob import TextBlob
import spacy

# Initialize spaCy model sentiment analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route("/", methods=["POST"])
def review_analyze(request: Request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json or 'review_text' not in request_json:
            logging.error("Missing 'review_text' in the request")
            return jsonify({"error": "Invalid input. 'review_text' is required."}), 400
        
        review_text = request_json.get('review_text')
        review_rating = request_json.get('review_rating')  # Optional

        # 1. Sentiment Analysis using TextBlob
        blob = TextBlob(review_text)
        sentiment_polarity = blob.sentiment.polarity
        if sentiment_polarity > 0:
            overall_sentiment = "positive"
        elif sentiment_polarity < 0:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        # 2. Aspect-Based Sentiment Analysis using spaCy
        aspect_sentiments = {}
        doc = nlp(review_text)
        for chunk in doc.noun_chunks:
            aspect = chunk.text
            aspect_sentiment = TextBlob(chunk.sent.text).sentiment.polarity
            if aspect_sentiment > 0:
                aspect_sentiments[aspect] = "positive"
            elif aspect_sentiment < 0:
                aspect_sentiments[aspect] = "negative"
            else:
                aspect_sentiments[aspect] = "neutral"

        # 3. Key Phrase Extraction using spaCy
        key_phrases = [chunk.text for chunk in doc.noun_chunks]

        # 4. Infer Rating (if not provided)
        inferred_rating = None
        if review_rating is None:
            inferred_rating = round((sentiment_polarity + 1) * 2.5)  # Scale -1 to 1 polarity to 1-5 rating

        # Build the response
        response = {
            "overall_sentiment": overall_sentiment,
            "aspect_sentiments": aspect_sentiments,
            "key_phrases": key_phrases,
        }
        if review_rating is not None:
            response["provided_rating"] = review_rating
        else:
            response["inferred_rating"] = inferred_rating

        # Return the structured JSON response
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500


if __name__ == "__main__":
    app.run()