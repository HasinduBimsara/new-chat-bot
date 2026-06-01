from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Import our custom local model inference logic
try:
    from inference import generate_local_response
    LOCAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Local model inference not available. Error: {e}")
    LOCAL_MODEL_AVAILABLE = False

load_dotenv() # Load environment variables from .env

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


app = Flask(__name__)
CORS(app) # Enable CORS for React frontend

# Load the Multilingual NLP Model (supports Sinhala)
print("Loading NLP Model (this may take a few seconds on first run)...")
# 'paraphrase-multilingual-MiniLM-L12-v2' is small, fast and supports 50+ languages including Sinhala
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("NLP Model loaded successfully!")

# Path to the dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "chatbot-web", "src", "dataset.json")

def load_dataset():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/api/recommend', methods=['POST'])
def recommend_suggestions():
    data = request.json
    user_answers = data.get('answers', []) # Expected format: [{"questionId": 1, "score": 3, "suggestions": [...]}, ...]
    chat_history = data.get('chatHistory', []) # [{sender: 'user', text: '...'}, ...]
    
    if not user_answers:
        return jsonify({"error": "No answers provided"}), 400
    
    # 1. Collect all raw suggestions from user's answers and their associated scores
    raw_suggestions = []
    has_severe = False
    for ans in user_answers:
        score = ans.get('score', 1)
        if score >= 4:
            has_severe = True
        suggestions = ans.get('suggestions', [])
        for sug in suggestions:
            raw_suggestions.append(sug)
            
    unique_texts = list(set(raw_suggestions))
    
    # 2. Extract key points from chat history
    chat_transcript = ""
    for msg in chat_history:
        sender_name = "රෝගියා" if msg['sender'] == 'user' else "උපදේශක"
        chat_transcript += f"{sender_name}: {msg['text']}\n"
        
    # 3. Use Local SLM to generate an initial summary of the questionnaire
    slm_questionnaire_summary = ""
    if LOCAL_MODEL_AVAILABLE:
        try:
            # Collect severe questions
            q_summary = "මට පහත ගැටලු ඇත: " + ", ".join([ans['question'] for ans in user_answers if ans.get('score', 1) >= 3])
            if q_summary == "මට පහත ගැටලු ඇත: ":
                q_summary = "මට විශේෂ ගැටලුවක් නැත."
            slm_questionnaire_summary = generate_local_response(system_context="ඔබ උපදේශකයෙකි.", user_message=q_summary, max_new_tokens=50)
        except Exception as e:
            print("SLM generation failed in recommendation:", e)
            slm_questionnaire_summary = "SLM දෝෂයකි."

    # 4. Use Gemini to generate a combined, intelligent summary
    if os.environ.get("GEMINI_API_KEY"):
        try:
            gen_model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""ඔබ දක්ෂ සිංහල මනෝවිද්‍යා උපදේශකයෙකි. 
පහත දැක්වෙන්නේ අපගේ Local AI Model (SLM) එක මගින් ප්‍රශ්නාවලියක් මත පදනම්ව ලබා දුන් මූලික නිගමනය, අදාළ සායනික යෝජනා (Clinical Suggestions) සහ රෝගියා සමග පැවැත්වූ කතාබහකි (Chat Transcript).

Local SLM හි මූලික නිගමනය (Questionnaire Analysis):
{slm_questionnaire_summary}

ප්‍රශ්නාවලියේ මූලික යෝජනා (Clinical Database):
{chr(10).join(unique_texts[:15])}

රෝගියා සමග කතාබහ:
{chat_transcript}

කරුණාකර මෙම දත්ත සියල්ල (SLM නිගමනය, මූලික යෝජනා සහ Chat එක) විශ්ලේෂණය කර, රෝගියාට වඩාත්ම ගැලපෙන, ප්‍රායෝගික මානසික සෞඛ්‍ය යෝජනා 10 ක් (හෝ ඊට වැඩි ගණනක්) සිංහලෙන් ලැයිස්තුගත කරන්න.
යෝජනා පමණක් ලබා දෙන්න (අංක යොදා). වෙනත් කතා අවශ්‍ය නැත."""
            
            response = gen_model.generate_content(prompt)
            # Parse the response text into a list
            suggestions = [line.strip() for line in response.text.split('\n') if line.strip()]
            
            # Clean up numbering (e.g. "1. යෝජනාව" -> "යෝජනාව")
            cleaned_suggestions = []
            for s in suggestions:
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', s).strip()
                if cleaned.startswith('*') or cleaned.startswith('-'):
                    cleaned = cleaned[1:].strip()
                if cleaned:
                    cleaned_suggestions.append(cleaned)
                    
            return jsonify({
                "total_raw": len(raw_suggestions),
                "total_filtered": len(cleaned_suggestions),
                "suggestions": cleaned_suggestions
            })
        except Exception as e:
            print("Gemini Recommendation Error:", str(e))
            # Fallback to simple list if Gemini fails
            pass
            
    # Fallback: Just return unique suggestions if Gemini is not available
    return jsonify({
        "total_raw": len(raw_suggestions),
        "total_filtered": len(unique_texts),
        "suggestions": unique_texts[:15] # Return up to 15
    })

@app.route('/api/chat', methods=['POST'])
def chat_with_bot():
    data = request.json
    user_message = data.get('message', '')
    context_history = data.get('contextHistory', []) 
    chat_history_raw = data.get('chatHistory', []) # From React: [{sender: 'user', text: '...'}, {sender: 'bot', text: '...'}]
    use_local_model = data.get('use_local_model', True) # Default to true for our custom model
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
        
        
    if not use_local_model and not os.environ.get("GEMINI_API_KEY"):
        return jsonify({"reply": "තාක්ෂණික දෝෂයක්: කරුණාකර Backend එකේ .env ගොනුවට GEMINI_API_KEY එක ඇතුළත් කරන්න."})

    # --- RAG Context Retrieval ---
    query_embedding = model.encode([user_message])
    dataset_dict = load_dataset()
    all_suggestions = []
    for q in dataset_dict['questionnaire']:
        for opt in q['options']:
            all_suggestions.extend(opt['suggestions'])
            
    all_suggestions = list(set(all_suggestions))
    doc_embeddings = model.encode(all_suggestions)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-3:][::-1]
    retrieved_context = [all_suggestions[i] for i in top_indices if similarities[i] > 0.4]

    # --- Construct System Instructions ---
    system_instruction = """You are a highly empathetic, professional Sri Lankan Psychological Counselor Agent.
You must communicate ENTIRELY in Sinhala (සිංහල).
Your role is to act as an Agentic Therapist. 

You must follow these 4 steps in your responses to guide the patient:
1. Empathy & Validation: Acknowledge the user's feelings without judgment.
2. CBT Reframing / Reassurance: Help the user identify negative thoughts or offer a reassuring perspective.
3. Actionable Advice (RAG): Provide ONE practical step or technique based on the provided context.
4. Probing Question: ALWAYS end your response with a gentle, open-ended question to dig deeper into the root cause of their feelings and keep the conversation going.

CRITICAL EMERGENCY GUARDRAIL: If the user mentions self-harm, suicide, or severe danger, STOP the standard process. Immediately urge them to contact the 1926 National Mental Health Helpline or Sri Lanka Sumithrayo in a highly compassionate tone.

--- PATIENT QUESTIONNAIRE PROFILE ---
"""
    has_severe = False
    for item in context_history:
        if item.get('score', 0) >= 3:
            system_instruction += f"- Problem: {item['question']} | Severity: {item['score']}/5\n"
            has_severe = True
            
    if not has_severe:
        system_instruction += "Patient is generally healthy.\n"
        
    system_instruction += "\n--- CLINICAL SUGGESTIONS (RAG Context) ---\n"
    if retrieved_context:
        for ctx in retrieved_context:
            system_instruction += f"- {ctx}\n"
    else:
        system_instruction += "No specific RAG context needed for this query.\n"

    # --- Format Chat History for Gemini ---
    formatted_history = []
    # We skip the last message in chat_history_raw because it is the current user_message which we will send via chat.send_message()
    for msg in chat_history_raw[:-1]:
        role = "user" if msg['sender'] == "user" else "model"
        formatted_history.append({
            "role": role,
            "parts": [msg['text']]
        })

    try:
        if use_local_model and LOCAL_MODEL_AVAILABLE:
            # --- LOCAL SLM GENERATION ---
            # Summarize the system context for the smaller local model
            local_context = "ඔබ දක්ෂ සිංහල මනෝවිද්‍යා උපදේශකයෙකි. රෝගියාට කරුණාවෙන් උපදෙස් දෙන්න."
                
            response_text = generate_local_response(system_context=local_context, user_message=user_message)
            
            # Since the 30M SLM might struggle to provide detailed counseling, 
            # we explicitly append the intelligent RAG context so the user gets a helpful answer.
            if retrieved_context:
                response_text = f"{response_text}\n\n💡 මානසික සෞඛ්‍ය යෝජනාව (RAG Context):\n{retrieved_context[0]}"
            
            return jsonify({
                "reply": response_text,
                "retrieved_context_used": len(retrieved_context),
                "model_used": "local_slm"
            })
            
        else:
            # --- GEMINI GENERATION (Fallback/Alternative) ---
            # Initialize Agentic Model
            gen_model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=system_instruction
            )
            
            # Start chat with memory
            chat = gen_model.start_chat(history=formatted_history)
            response = chat.send_message(user_message)
            
            return jsonify({
                "reply": response.text.strip(),
                "retrieved_context_used": len(retrieved_context),
                "model_used": "gemini"
            })
    except Exception as e:
        print("LLM Error:", str(e))
        return jsonify({"reply": "සමාවෙන්න, මාගේ බුද්ධිමය පද්ධතියේ දෝෂයක්. කරුණාකර පසුව නැවත උත්සාහ කරන්න."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
