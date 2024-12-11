import os
import json
import logging

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
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_MODEL = "gpt-4o"

# Initialize OpenAI client
openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
logging.basicConfig(level=logging.DEBUG)

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
    """Get chat history and ensure proper session initialization."""
    if 'chat_history' not in session:
        session['chat_history'] = []
        session['patient_summary'] = "No symptoms described yet."
    return session['chat_history']

def get_messages_for_openai():
    """Construct messages array for OpenAI with proper context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history
    for msg in session.get('chat_history', []):
        messages.append({
            "role": "user" if msg['type'] == 'user' else "assistant",
            "content": msg['content']
        })
    
    # Add patient summary if available
    if session.get('patient_summary'):
        messages.append({
            "role": "system",
            "content": f"Current patient information: {session['patient_summary']}"
        })
    
    return messages

def count_assistant_questions(messages):
    """Count how many questions the assistant has asked."""
    return sum(1 for m in messages if m['role'] == 'assistant' and '?' in m['content'])

def update_patient_summary(message):
    """Update the patient summary based on the user's message."""
    current_summary = session.get('patient_summary', "No symptoms described yet.")
    
    # Simple keyword detection
    keywords = {
        'pain': 'pain',
        'ache': 'aching',
        'tooth': 'tooth',
        'teeth': 'teeth',
        'gum': 'gums',
        'jaw': 'jaw',
        'mouth': 'mouth'
    }
    
    new_info = []
    message_lower = message.lower()
    
    for keyword, description in keywords.items():
        if keyword in message_lower:
            new_info.append(description)
    
    if new_info:
        if current_summary == "No symptoms described yet.":
            current_summary = ""
        new_summary = f"{current_summary} Reports {', '.join(new_info)}."
        session['patient_summary'] = new_summary.strip()
        return new_summary
    
    return current_summary

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

def get_finalization_response(patient_summary):
    """Make a separate API call to get the final recommendation."""
    # Build a new messages array for finalization
    final_messages = []
    
    # System prompt for finalization
    final_messages.append({"role": "system", "content": f"""
The user has answered 3 questions.

Known info:
{patient_summary}

Conditions and Appointment Types:
{conditions_summary}
{appointment_summary}

Instructions:
1. Pick the most likely condition or fallback.
2. Recommend appointment type.
3. If info_card_id, provide link.
4. Conclude: 'It seems like you need an appointment for ... Here are all available times...'

Do not ask questions now. Just finalize."""})

    # Call the model fresh with no history
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=final_messages,
        temperature=0.0,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            logging.warning("Empty message received")
            return jsonify({'error': 'Empty message'}), 400

        logging.info(f"Received message: {user_message}")

        # Get chat history and update patient summary
        chat_history = get_chat_history()
        patient_summary = update_patient_summary(user_message)
        logging.debug(f"Updated patient summary: {patient_summary}")
        
        # Get messages with context
        messages = get_messages_for_openai()
        messages.append({"role": "user", "content": user_message})
        
        logging.debug(f"Sending messages to OpenAI: {messages}")
        
        # Get response from OpenAI
        try:
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=0.7  # More natural responses
            )
            
            ai_response = response.choices[0].message.content
            logging.info(f"Received AI response: {ai_response}")
            
        except Exception as api_error:
            logging.error(f"OpenAI API error: {str(api_error)}")
            return jsonify({'error': 'Failed to get AI response'}), 500
        
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
        logging.error(f"General error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Running on port 5000 for Replit compatibility
        logging.info("Starting Flask server on port 5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logging.error(f"Failed to start Flask server: {str(e)}")
        raise
