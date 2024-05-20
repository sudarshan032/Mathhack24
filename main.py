from transformers import pipeline
import os
from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set your API key here or use an environment variable
api_key = os.getenv('openai')
aai.settings.api_key = os.getenv('assemblyai')
client = OpenAI(api_key=api_key)

# Initialize a dictionary to store conversation history
conversation_history = {}

@app.route('/')
def home():
    return "Welcome to the OpenAI Chat API. Use the /chat endpoint to interact."

@app.route('/texttotext', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    user_id = request.json.get('user_id')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    if not user_id:
        return jsonify({'error': 'No user_id provided'}), 400

    # Initialize conversation history for the user if not already present
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Append the user's message to the conversation history
    conversation_history[user_id].append({"role": "user", "content": user_message})

    # Generate response using OpenAI's API
    chat_completion = client.chat.completions.create(
        messages=conversation_history[user_id],
        model="gpt-3.5-turbo",
    )

    # Get the AI's response and append it to the conversation history
    ai_response = chat_completion.choices[0].message.content
    conversation_history[user_id].append({"role": "assistant", "content": ai_response})

    return jsonify({'response': ai_response})

@app.route('/audioemotion', methods=['POST'])
def audio_chat():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(filename)

        # Load the audio classification pipeline
        pipe = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

        # Get the predicted emotion
        result = pipe(filename)

        # Sort the results by score in descending order
        result = sorted(result, key=lambda x: x['score'], reverse=True)

        return jsonify(result)

@app.route('/tts', methods=['POST'])
def text_to_speech():
    user_message = request.json.get('audio_message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=user_message
    )
    response.stream_to_file(speech_file_path)

    return jsonify({'file_path': str(speech_file_path)})

@app.route('/stt', methods=['POST'])
def speech_to_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(filename)
        return jsonify(transcript.text)

if __name__ == '__main__':
    app.run(debug=True)
