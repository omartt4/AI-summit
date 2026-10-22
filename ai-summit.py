
import os
from dotenv import load_dotenv
from google import genai
import google.genai.types as types

# Load API key from .env
load_dotenv()

api_key = os.getenv("gemini_api")
model = os.getenv("model")
temperature = float(os.getenv("temperature"))
max_output_tokens = int(os.getenv("max_output_tokens"))

if not api_key:
    raise ValueError("gemini_api was not found in .env")

if not model:
    raise ValueError("model was not found in .env")
if not temperature:
    raise ValueError("temperature was not found in .env")

if not max_output_tokens:
    raise ValueError("max_output_tokens was not found in .env")


# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Define your custom configuration
config = types.GenerateContentConfig(
    # Persona & steering
    system_instruction="You are a strict, concise technical assistant. summarize in 2 sentences or less.",
    
    # Sampling controls
    temperature=temperature,
    max_output_tokens=max_output_tokens,
    
    
    # Optional: Structured JSON Output
    # response_mime_type="application/json",
)

# Send a request
response = client.models.generate_content(
    model=model,
    config=config,
    contents="stop_sequences are specific strings you define that tell the model to instantly halt text generation once predicted, making them highly useful for enforcing strict formatting or preventing the model from generating unwanted conversational turns. Meanwhile, presence_penalty and frequency_penalty are parameters used to control repetition and diversity by penalizing words that have already appeared in the output; presence_penalty applies a one-time flat tax the moment a word is used to encourage the introduction of novel topics, whereas frequency_penalty applies an escalating tax based on the exact number of times a word has appeared, forcing the model to expand its vocabulary and avoid getting stuck in repetitive loops."
)

print(response.text)


