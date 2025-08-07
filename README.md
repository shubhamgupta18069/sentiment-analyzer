# Cloud-Based Sentiment Analyzer Using Amazon Comprehend

## Overview
This project performs sentiment analysis using Amazon Comprehend, AWS's fully managed NLP service. It takes input from a text file, detects sentiment, and stores the result as JSON.

## AWS Services Used
- Amazon Comprehend
- (Optional) Amazon S3

## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Configure AWS credentials:
    ```
    aws configure
    ```

3. Run the app:
    ```
    python app.py
    ```

4. (Optional) Upload results to S3:
    ```
    python upload_to_s3.py
    ```

## File Descriptions
- `app.py` - Main logic to analyze sentiment.
- `input.txt` - Text to analyze.
- `output.json` - Result from Comprehend.
- `upload_to_s3.py` - Uploads input/output to S3.
- `requirements.txt` - Python dependencies.

## Report Line
"This project demonstrates a simple cloud-based NLP workflow using Amazon Comprehend, integrated with AWS S3 for storage and optionally deployable on AWS EC2."

