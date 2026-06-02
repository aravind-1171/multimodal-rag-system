import boto3
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\Users\USER\multimodal-rag\.env")

s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
BUCKET = os.getenv('AWS_BUCKET_NAME')

def upload_file(file_path):
    file_name = os.path.basename(file_path)
    s3_key = f"docs/{file_name}"
    s3.upload_file(file_path, BUCKET, s3_key)
    print(f"✅ Uploaded: {file_name} → s3://{BUCKET}/{s3_key}")
    return file_name

def list_files():
    response = s3.list_objects_v2(Bucket=BUCKET)
    files = [obj['Key'] for obj in response.get('Contents', [])]
    print("📁 Files in bucket:", files)
    return files

def download_file(file_name, save_path):
    s3.download_file(BUCKET, file_name, save_path)
    print(f"⬇️ Downloaded: {file_name} → {save_path}")

# Test it
if __name__ == "__main__":
    # Upload a test file
    with open("test.txt", "w") as f:
        f.write("Hello from multimodal RAG!")
    
    upload_file("test.txt")
    list_files()