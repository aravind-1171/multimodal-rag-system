import json
import boto3
import os
import sys
import pickle

s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='ap-south-1')

def get_embedding(text):
    body = json.dumps({"inputText": text[:8000]})
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    result = json.loads(response['body'].read())
    return result['embedding']

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def classify_document(file_name):
    ext = os.path.splitext(file_name)[1].lower()
    if ext == '.csv':
        return 'tabular'
    elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
        return 'image'
    else:
        return 'text/pdf'

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        # Skip index files
        if key.startswith('indexes/') or key.startswith('lambda'):
            return {'statusCode': 200, 'body': 'Skipped'}

        print(f"📄 Processing: {key} from {bucket}")

        local_path = f"/tmp/{os.path.basename(key)}"
        s3.download_file(bucket, key, local_path)
        print(f"✅ Downloaded to {local_path}")

        doc_type = classify_document(key)
        print(f"📊 Document type: {doc_type}")

        if key.endswith('.pdf'):
            import pypdf
            reader = pypdf.PdfReader(local_path)
            raw_text = "\n".join(
                page.extract_text() for page in reader.pages
            )
            chunks = chunk_text(raw_text)

        elif key.endswith('.csv'):
            import csv
            chunks = []
            with open(local_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    chunks.append(f"Columns: {', '.join(rows[0].keys())}, Total rows: {len(rows)}")
                for row in rows:
                    chunks.append(" | ".join([f"{k}: {v}" for k, v in row.items()]))

        else:
            with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            chunks = chunk_text(raw_text)

        print(f"✅ Created {len(chunks)} chunks")

        # Generate embeddings
        print("⏳ Generating embeddings...")
        embeddings = [get_embedding(chunk) for chunk in chunks]

        # Save to S3
        index_key = f"indexes/{os.path.basename(key)}_chunks.pkl"
        embeddings_key = f"indexes/{os.path.basename(key)}_embeddings.pkl"

        s3.put_object(
            Bucket=bucket,
            Key=index_key,
            Body=pickle.dumps(chunks)
        )
        s3.put_object(
            Bucket=bucket,
            Key=embeddings_key,
            Body=pickle.dumps(embeddings)
        )

        print(f"✅ Saved index to S3")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Success',
                'file': key,
                'doc_type': doc_type,
                'chunks': len(chunks)
            })
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}