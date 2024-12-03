import os
import re
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-default-secret-key")

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_MODEL = "gpt-4o"

# Initialize OpenAI client
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a dental AI assistant that helps patients identify possible dental issues based on their symptoms.
Rules:
- Keep responses under 3 sentences unless follow-up questions are needed
- Use simple, empathetic language
- Reference info cards using [InfoCard: CardName] format
- Never provide definitive diagnoses
- Be direct and concise

Common situations and responses:
For tooth pain: Ask about pain type (sharp/dull) and triggers (hot/cold/pressure)
For bleeding gums: Inquire about brushing habits and last dental visit
For cosmetic concerns: Ask about specific aesthetic goals
For emergencies: Emphasize seeking immediate professional care
"""

def get_chat_history():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def detect_placeholders(text):
    # Detect [InfoCard: Name] and [3DModel: Name] patterns
    info_cards = re.findall(r'\[InfoCard:\s*([^\]]+)\]', text)
    models = re.findall(r'\[3DModel:\s*([^\]]+)\]', text)
    return info_cards, models

@app.route('/')
def home():
    # Clear session when starting new chat
    session['chat_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        chat_history = get_chat_history()
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add chat history
        for msg in chat_history:
            messages.append({
                "role": "user" if msg['type'] == 'user' else "assistant",
                "content": msg['content']
            })
            
        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Get response from OpenAI
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=250,
            temperature=0.5
        )

        ai_response = response.choices[0].message.content
        
        # Detect any placeholders in the response
        info_cards, models = detect_placeholders(ai_response)
        
        # Update chat history
        chat_history.append({"type": "user", "content": user_message})
        chat_history.append({"type": "assistant", "content": ai_response})
        session['chat_history'] = chat_history

        return jsonify({
            "response": ai_response,
            "info_cards": info_cards,
            "models": models
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
