import os
import json

# Load JSON data files
with open('data/appointment_types.json') as f:
    appointment_types_data = json.load(f)

with open('data/conditions.json') as f:
    conditions_data = json.load(f)

with open('data/info_cards.json') as f:
    info_cards_data = json.load(f)

with open('data/clinic_config.json') as f:
    clinic_config_data = json.load(f)

# Build dictionaries for quick lookup
appointment_types_map = {at['id']: at for at in appointment_types_data['appointment_types']}
conditions_map = {c['id']: c for c in conditions_data['conditions']}

def get_appointment_duration(condition_id):
    """
    Calculate the final appointment duration considering condition requirements and clinic overrides.
    """
    cond = conditions_map[condition_id]
    at_id = cond['appointment_type_id']
    base_duration = appointment_types_map[at_id]['default_duration_minutes']

    # Add condition-level extra time if any
    extra = cond.get('additional_time_minutes', 0)
    duration = base_duration + extra

    # Apply clinic overrides
    # Check if appointment type overridden
    if at_id in clinic_config_data.get('appointment_type_overrides', {}):
        duration = clinic_config_data['appointment_type_overrides'][at_id]
    
    # Check condition overrides
    if condition_id in clinic_config_data.get('condition_overrides', {}):
        duration = clinic_config_data['condition_overrides'][condition_id]

    return duration

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

# Build summaries
conditions_summary = "Conditions:\n"
for c in conditions_data['conditions']:
    conditions_summary += f"- {c['id']}: {c['name_en']} -> {c['appointment_type_id']}\n"

appointment_summary = "Appointment Types:\n"
for at in appointment_types_data['appointment_types']:
    appointment_summary += f"- {at['id']}: {at['name_en']} (Default: {at['default_duration_minutes']}min)\n"

# Load system prompt from file
with open('prompts/system_prompt.txt') as f:
    base_system_prompt = f.read()

SYSTEM_PROMPT = base_system_prompt + "\n\n" + conditions_summary + "\n" + appointment_summary

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
            temperature=0.0  # Using 0.0 for more deterministic behavior
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
