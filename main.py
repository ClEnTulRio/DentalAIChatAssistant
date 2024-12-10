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
    # Initialize all session variables if not present
    if 'chat_history' not in session:
        session.clear()  # Clear any stale session data
        session['chat_history'] = []
        session['questions_asked'] = 0
        session['patient_summary'] = "No symptoms described yet."
        session['finalization_done'] = False
        print("DEBUG: New session initialized with questions_asked =", session['questions_asked'])
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
            return jsonify({'error': 'Empty message'}), 400

        # Get or initialize chat history and session state
        chat_history = get_chat_history()
        questions_asked = session.get('questions_asked', 0)
        finalization_done = session.get('finalization_done', False)
        
        print(f"DEBUG: Current state - questions_asked: {questions_asked}, finalization_done: {finalization_done}")
        
        # Update patient summary with new information
        patient_summary = update_patient_summary(user_message)
        print(f"DEBUG: Updated patient summary: {patient_summary}")
        
        # If we've already asked 3 questions or finalization is done, always use finalization response
        if questions_asked >= 3 or finalization_done:
            print("DEBUG: Entering finalization phase")
            if not finalization_done:
                session['finalization_done'] = True
                print("DEBUG: Marked finalization as done")
            ai_response = get_finalization_response(patient_summary)
        else:
            # Normal Q&A phase
            print(f"DEBUG: Entering Q&A phase (question {questions_asked + 1}/3)")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            # Add chat history
            for msg in chat_history:
                messages.append({
                    "role": "user" if msg['type'] == 'user' else "assistant",
                    "content": msg['content']
                })
                
            # Add current message and context
            messages.append({"role": "user", "content": user_message})
            messages.append({
                "role": "system",
                "content": f"Current known information: {patient_summary}"
            })
            
            # Add question count context
            remaining_questions = 3 - questions_asked
            messages.append({
                "role": "system",
                "content": f"You have asked {questions_asked} questions so far. "
                          f"You may ask {remaining_questions} more question(s) if needed."
            })
            
            # Get response from OpenAI
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=250,
                temperature=0.0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            ai_response = response.choices[0].message.content
            
            # Check if the response contains a question
            if '?' in ai_response and not session.get('finalization_done', False):
                session['questions_asked'] = session.get('questions_asked', 0) + 1
                print(f"DEBUG: Incremented questions_asked to {session['questions_asked']}")
                
                # If this was the third question, next response will be finalization
                if session['questions_asked'] >= 3:
                    print("DEBUG: Third question asked, next response will be finalization")
        
        # Detect any placeholders in the response
        info_cards, models = detect_placeholders(ai_response)
        
        # Update chat history
        chat_history.append({"type": "user", "content": user_message})
        chat_history.append({"type": "assistant", "content": ai_response})
        session['chat_history'] = chat_history
        
        print(f"DEBUG: Returning response. questions_asked: {session.get('questions_asked', 0)}, finalization_done: {session.get('finalization_done', False)}")
        
        return jsonify({
            "response": ai_response,
            "info_cards": info_cards,
            "models": models,
            "debug_info": {
                "questions_asked": session.get('questions_asked', 0),
                "finalization_phase": session.get('finalization_done', False)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
