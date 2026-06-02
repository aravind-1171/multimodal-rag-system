import base64
import os
from groq import Groq

import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"C:\Users\USER\multimodal-rag\.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def encode_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_text_from_image(image_path):
    """
    Sends image to Groq vision model
    and gets back a detailed text description
    """
    print(f"🖼️ Processing image: {os.path.basename(image_path)}")

    # Get image extension
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    media_type = media_types.get(ext, "image/jpeg")

    # Encode image
    image_data = encode_image(image_path)

    # Send to Groq vision
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """Analyze this image in detail. Include:
1. What the image shows overall
2. Any text visible in the image
3. Any numbers, data, or statistics
4. Key objects or elements
5. Any charts, graphs, or diagrams explained in detail
Be as descriptive as possible so someone can answer questions about this image."""
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    description = response.choices[0].message.content
    print(f"✅ Image analyzed successfully")
    print(f"📝 Description preview: {description[:200]}...")
    return [description]


# ── Test ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with any image in your project folder
    # Create a simple test image first
    from PIL import Image, ImageDraw

    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "RAG System Test Image", fill='black')
    draw.rectangle([20, 20, 380, 180], outline='blue', width=3)
    img.save("test_image.png")
    print("✅ Test image created")

    # Extract text from image
    chunks = extract_text_from_image("test_image.png")
    print(f"\n📄 Extracted description:\n{chunks[0]}")