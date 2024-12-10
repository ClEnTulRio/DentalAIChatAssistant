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
        session['questions_asked'] = 0
        session['patient_summary'] = "No symptoms described yet."
    return session['chat_history']

def count_assistant_questions(messages):
    """Count how many questions the assistant has asked."""
    return sum(1 for m in messages if m['role'] == 'assistant' and '?' in m['content'])

def update_patient_summary(message):
    """Update the patient summary based on the user's message."""
    summary = session.get('patient_summary', "")
    
    # Define keywords to look for
    symptoms = {
        'pain': 'reports pain',
        'ache': 'reports aching',
        'hurt': 'reports pain',
        'sensitive': 'reports sensitivity',
        'swollen': 'reports swelling',
        'bleeding': 'reports bleeding',
        'red': 'reports redness',
        'loose': 'reports looseness',
        'broken': 'reports broken/damaged tooth',
        'chip': 'reports chipped tooth',
        'crack': 'reports cracked tooth'
    }
    
    # Define locations
    locations = {
        'tooth': 'in tooth',
        'teeth': 'in teeth',
        'gum': 'in gums',
        'jaw': 'in jaw',
        'mouth': 'in mouth',
        'face': 'in face'
    }
    
    # Parse message for symptoms and locations
    message_lower = message.lower()
    detected_symptoms = []
    for keyword, description in symptoms.items():
        if keyword in message_lower:
            detected_symptoms.append(description)
    
    detected_locations = []
    for keyword, description in locations.items():
        if keyword in message_lower:
            detected_locations.append(description)
    
    # Combine findings into a summary
    new_findings = []
    if detected_symptoms:
        new_findings.extend(detected_symptoms)
    if detected_locations:
        new_findings.extend(detected_locations)
    
    if new_findings:
        if summary == "No symptoms described yet.":
            summary = ""
        new_summary = f"{summary} Patient {', '.join(new_findings)}."
        session['patient_summary'] = new_summary.strip()
        return new_summary
    
    return summary

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
            
        # Update patient summary with new information from user message
        patient_summary = update_patient_summary(user_message)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Add the current patient summary as context
        messages.append({
            "role": "system",
            "content": f"Current known information: {patient_summary}"
        })
        
        # Check how many questions have been asked so far
        questions_asked = count_assistant_questions(messages)
        
        # Add appropriate system message based on question count
        if questions_asked < 3:
            remaining_questions = 3 - questions_asked
            messages.append({
                "role": "system",
                "content": f"You have asked {questions_asked} questions so far. You may ask {remaining_questions} more question(s). "
                          f"DO NOT ask about information already provided in the summary above. "
                          "After your questions are done, you must finalize without asking more questions."
            })
        else:
            messages.append({
                "role": "system",
                "content": "You have asked 3 questions total. NOW FINALIZE. Do NOT ask another question. "
                          "Based on the summary above, pick condition, recommend appointment type, "
                          "provide info card link if available, and say 'It seems like you need...'. "
                          "Breaking this rule is not acceptable."
            })

        # Get response from OpenAI
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=250,
            temperature=0.0,  # Using 0.0 for more deterministic behavior
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
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
