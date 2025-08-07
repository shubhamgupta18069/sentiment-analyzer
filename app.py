from flask import Flask, request, render_template
import boto3
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS config
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION', 'ap-south-1')
endpoint_name = 'sentiment-endpoint'  # Deployed SageMaker endpoint

# Create SageMaker runtime client
if aws_key and aws_secret:
    runtime = boto3.client(
        'sagemaker-runtime',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )
else:
    runtime = boto3.client('sagemaker-runtime', region_name=aws_region)

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
                # Call SageMaker endpoint
                response = runtime.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps({"inputs": text})
                )
                result = json.loads(response['Body'].read().decode())

                # Parse SageMaker response
                sentiment = result[0]['label']
                score = result[0]['score']
                scores = {sentiment: score}

            except Exception as e:
                print("ERROR:", str(e))
                return f"<h3>Error: {str(e)}</h3>", 500
    return render_template('index.html', sentiment=sentiment, scores=scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
