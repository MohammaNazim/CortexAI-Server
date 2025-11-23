import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file (should contain GEMINI_API_KEY)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env")
    exit(1)

try:
    # Configure Gemini API
    genai.configure(api_key=api_key)

    print("🔹 Listing available Gemini models...\n")

    models = genai.list_models()

    for m in models:
        print(f"✅ {m.name} — supports: {m.supported_generation_methods}")

    print("\n🎯 Done. Your Gemini API key is working correctly.")

except Exception as e:
    print("❌ Failed to connect to Gemini API:")
    print(e)
