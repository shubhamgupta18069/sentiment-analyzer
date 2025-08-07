from flask import Flask, request, render_template
import boto3
import os
from dotenv import load_dotenv

# Try loading environment variables
load_dotenv()

# Check if .env has AWS keys
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')

# If both key and secret exist, use them; else rely on IAM role (boto3 default)
if aws_key and aws_secret:
    comprehend = boto3.client(
        'comprehend',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )
else:
    comprehend = boto3.client('comprehend', region_name=aws_region)

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    scores = None
    if request.method == 'POST':
        text = request.form['text']
        if text:
            try:
                response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
                sentiment = response['Sentiment']
                scores = response['SentimentScore']
            except Exception as e:
                print("ERROR:", str(e))
                return f"<h3>Error: {str(e)}</h3>", 500
    return render_template('index.html', sentiment=sentiment, scores=scores)
if __name__ == '__main__':
    app.run(host='0.0.0.0')
