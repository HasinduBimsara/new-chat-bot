from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import random
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
        
    # Calculate severity and dynamic limits
    max_score = 1
    if user_answers:
        max_score = max([ans.get('score', 1) for ans in user_answers])
    
    if max_score <= 2:
        max_suggestions_limit = 5  # Mild/Healthy — increased for more coverage
    elif max_score == 3:
        max_suggestions_limit = 7  # Moderate — more suggestions for moderate issues
    else:
        max_suggestions_limit = 8  # Severe — comprehensive but still focused
    
    # 1. Collect all raw suggestions from user's answers and their associated scores
    raw_suggestions = []
    has_severe = False
    dataset_dict = None
    
    # Filter questionnaire suggestions to only include matching severity
    # If the user has moderate or severe issues (max_score >= 3), we do not include suggestions for healthy/mild options (scores 1 and 2)
    if max_score >= 3:
        relevant_answers = [ans for ans in user_answers if ans.get('score', 1) >= 3]
    else:
        relevant_answers = user_answers

    for ans in relevant_answers:
        score = ans.get('score', 1)
        if score >= 4:
            has_severe = True
        suggestions = ans.get('suggestions', [])
        
        # If suggestions list is empty, look it up in dataset.json
        if not suggestions:
            if dataset_dict is None:
                try:
                    dataset_dict = load_dataset()
                except Exception as e:
                    print("Error loading dataset in recommendation lookup:", e)
            if dataset_dict:
                q_id = ans.get('questionId')
                # Find matching question in dataset questionnaire
                q_match = next((q for q in dataset_dict['questionnaire'] if q['id'] == q_id), None)
                if q_match:
                    opt_match = next((opt for opt in q_match['options'] if opt['score'] == score), None)
                    if opt_match:
                        suggestions = opt_match.get('suggestions', [])
                        
        for sug in suggestions:
            raw_suggestions.append(sug)
            
    unique_texts = list(set(raw_suggestions))
    # Shuffle for variety — each request gets a different ordering
    random.shuffle(unique_texts)
    
    # 2. Extract key points from chat history
    chat_transcript = ""
    user_chat_messages = []
    for msg in chat_history:
        sender_name = "රෝගියා" if msg['sender'] == 'user' else "උපදේශක"
        chat_transcript += f"{sender_name}: {msg['text']}\n"
        if msg['sender'] == 'user':
            user_chat_messages.append(msg['text'])
            
    # 2b. Retrieve suggestions related to user's chat messages using RAG (Highly accurate combined query)
    chat_suggestions = []
    if user_chat_messages:
        try:
            dataset_dict = load_dataset()
            all_db_suggestions = []
            for q in dataset_dict['questionnaire']:
                for opt in q['options']:
                    # Filter RAG suggestions: if user has issues, don't search through healthy options (score < 3)
                    if max_score >= 3 and opt['score'] < 3:
                        continue
                    # If user is healthy, don't retrieve severe recommendations
                    if max_score < 3 and opt['score'] >= 3:
                        continue
                    all_db_suggestions.extend(opt['suggestions'])
            all_db_suggestions = list(set(all_db_suggestions))
            
            # Encode database suggestions
            doc_embeddings = model.encode(all_db_suggestions)
            
            # Combine user messages to extract overall context and matching suggestions
            combined_user_chat = " ".join([m for m in user_chat_messages if len(m.strip()) >= 5])
            if combined_user_chat:
                query_embedding = model.encode([combined_user_chat])
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                # Match top 4 with a higher threshold (similarity > 0.42) for maximum accuracy
                top_indices = np.argsort(similarities)[-4:][::-1]
                for idx in top_indices:
                    if similarities[idx] > 0.42:
                        chat_suggestions.append(all_db_suggestions[idx])
        except Exception as e:
            print("Error retrieving fallback chat suggestions:", e)
            
    # Combine questionnaire suggestions and chat RAG suggestions
    combined_raw_suggestions = raw_suggestions + chat_suggestions
    unique_texts = list(set(combined_raw_suggestions))
    # Shuffle for variety — ensures different ordering on each request
    random.shuffle(unique_texts)
        
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
    # Generate a random seed phrase to force Gemini to vary its output each time
    variety_seed = random.randint(1000, 9999)
    variety_styles = [
        "මෙවර විශේෂයෙන්ම ප්‍රායෝගික ක්‍රියාමාර්ග කෙරෙහි අවධානය යොමු කරන්න.",
        "මෙවර හැඟීම් පාලනය සහ CBT ක්‍රමවේද කෙරෙහි වැඩි අවධානයක් යොමු කරන්න.",
        "මෙවර ශාරීරික සෞඛ්‍යය සහ දෛනික පුරුදු වෙනස් කිරීම කෙරෙහි අවධානය යොමු කරන්න.",
        "මෙවර සමාජ සම්බන්ධතා සහ පවුල් සහාය කෙරෙහි වැඩි අවධානයක් යොමු කරන්න.",
        "මෙවර මනසේ සන්සුන් බව සහ භාවනා/හුස්ම ගැනීමේ ක්‍රමවේද කෙරෙහි අවධානය යොමු කරන්න.",
    ]
    variety_instruction = random.choice(variety_styles)
    
    if os.environ.get("GEMINI_API_KEY"):
        try:
            gen_model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""ඔබ දක්ෂ සිංහල මනෝවිද්‍යා උපදේශකයෙකි. (Session ID: {variety_seed})
පහත දැක්වෙන්නේ අපගේ Local AI Model (SLM) එක මගින් ප්‍රශ්නාවලියක් මත පදනම්ව ලබා දුන් මූලික නිගමනය, ප්‍රශ්නාවලියෙන් ලැබුණු මූලික යෝජනා (Clinical Database) සහ රෝගියා සමග පැවැත්වූ අවසාන කතාබහකි (Chat Transcript).

Local SLM හි මූලික නිගමනය (Questionnaire Analysis):
{slm_questionnaire_summary}

ප්‍රශ්නාවලියේ මූලික යෝජනා (Clinical Database — මෙම ලැයිස්තුව සෑම වරම වෙනස් අනුපිළිවෙළකින් ලැබේ):
{chr(10).join(unique_texts[:25])}

රෝගියා සමග කතාබහ (Chat Transcript):
{chat_transcript}

කරුණාකර මෙම දත්ත සියල්ල (SLM නිගමනය, මූලික යෝජනා සහ Chat Transcript එක) ඉතා හොඳින් විශ්ලේෂණය කරන්න.

=== ප්‍රතිඵල ව්‍යුහය (RESPONSE STRUCTURE) ===
අවසාන ලැයිස්තුව පහත ක්‍රමයට සකස් කරන්න:

කොටස 1 — ප්‍රායෝගික/සාමාන්‍ය යෝජනා (FIRST — General/Practical Suggestions):
මුලින්ම, රෝගියාගේ තත්ත්වයට ගැලපෙන ප්‍රායෝගික, දෛනික ජීවිතයට අදාළ යෝජනා ලබාදෙන්න. උදාහරණ: ව්‍යායාම, නින්ද, ආහාර, භාවනා, හුස්ම ගැනීමේ ක්‍රම, journaling, ආදිය.

කොටස 2 — අනිවාර්ය/බරපතල යෝජනා (LAST — Compulsory/Critical Suggestions):
ලැයිස්තුවේ අවසානයට පමණක්, රෝගියාගේ බරපතලකම අනුව පහත වැනි අනිවාර්ය උපදෙස් එක් කරන්න:
- වෘත්තීය උපදේශකයෙකුගේ සහාය ලබාගන්න
- 1926 හෝ සුමිත්‍රයෝ සංවිධානය අමතන්න
- රෝහල් මනෝවෛද්‍ය ඒකකයකින් ප්‍රතිකාර ලබාගන්න
- ආදී බරපතල/අනිවාර්ය යෝජනා
(max score 1 හෝ 2 නම් මෙම කොටස අවශ්‍ය නැත. max score 3 නම් උපදේශකයෙකු හමුවීම පමණක්. max score 4 හෝ 5 නම් 1926, සුමිත්‍රයෝ, වෛද්‍ය සහාය ආදිය ඇතුළත් කරන්න.)

=== විශේෂ උපදෙස් ===
1. රෝගියා කතාබහේදී (Chat Transcript) ඉදිරිපත් කළ නව ගැටලු, හැඟීම්, සිතුවිලි සහ තොරතුරු කෙරෙහි විශේෂ අවධානයක් යොමු කරන්න.
2. අවසාන ලැයිස්තුවෙහි ඇති යෝජනා (Suggestions) ප්‍රශ්නාවලියේ මූලික යෝජනා (Clinical Database) වල ශෛලියට සහ ආකෘතියට (Short, direct, practical, single-sentence plain-text Sinhala advice) උපරිමයෙන් සමාන විය යුතුය.
3. කිසිම හේතුවක් මත markdown formatting, තරු ලකුණු (* හෝ **), bold text හෝ මාතෘකා (headers) භාවිත නොකරන්න. එක් එක් යෝජනාව තනි සරල වාක්‍යයකින් (Plain-text) පමණක් සකස් කරන්න. උදාහරණයක් ලෙස: 'නින්දට යාමට පෙර තිර (Screens) භාවිතය අඩු කරන්න.' වැනි කෙටි සෘජු වාක්‍ය පමණක් ලියන්න.
4. හැකිතාක් දුරට ලබා දී ඇති Clinical Database ලැයිස්තුවේ ඇති යෝජනා සෘජුවම තෝරාගෙන භාවිත කරන්න. රෝගියා කතාබහේදී සඳහන් කළ අලුත්ම කරුණු වෙනුවෙන් අලුතින්ම යෝජනා සාදන්නේ නම්, ඒවාද Clinical Database එකෙහි ඇති යෝජනා වල ශෛලියටම (Short, direct action items) ගැලපෙන සේ සකස් කරන්න.
5. {variety_instruction}
6. සෑම වරම වෙනස් යෝජනා combination එකක් තෝරාගන්න. පෙර වාරවල දුන් යෝජනාම නැවත නැවත නොදෙන්න. Clinical Database එකෙන් විවිධ යෝජනා තෝරාගෙන, නව වචන ප්‍රයෝග (synonyms/paraphrasing) භාවිතයෙන් යෝජනා සකස් කරන්න. එකම අදහස වෙනස් ආකාරයකින් ප්‍රකාශ කරන්න.
7. අවසාන යෝජනා ගණන රෝගියාගේ මානසික සෞඛ්‍යයේ බරපතලකම (max score = {max_score}/5) මත පදනම්ව තීරණය කරන්න:
   - රෝගියාගේ ගැටලු ඉතා අවම නම් (max score 1 හෝ 2), ලැයිස්තුවේ ප්‍රායෝගික උපදෙස් 4-5 ක් පමණි.
   - රෝගියාගේ ගැටලු මධ්‍යස්ථ මට්ටමේ නම් (max score 3), ලැයිස්තුවේ ප්‍රායෝගික උපදෙස් 4-5 ක් + අනිවාර්ය උපදෙස් 1-2 ක් (මුළු 5-7 ක්) තිබිය යුතුය.
   - රෝගියාගේ ගැටලු බරපතල මට්ටමේ නම් (max score 4 හෝ 5), ලැයිස්තුවේ ප්‍රායෝගික උපදෙස් 5-6 ක් + අනිවාර්ය උපදෙස් 2-3 ක් (මුළු 7-8 ක්) තිබිය යුතුය.
   කිසිම හේතුවක් මත මුළු යෝජනා ගණන {max_suggestions_limit} සීමාව ඉක්මවා නොයන්න. අනවශ්‍ය හෝ නැවත නැවත කියවෙන උපදෙස් ලැයිස්තුගත කිරීමෙන් වළකින්න.
8. එකිනෙකට සමාන හෝ එකම අර්ථය දෙන යෝජනා දෙකක් ලැයිස්තුවේ ඇතුළත් නොකරන්න. සෑම යෝජනාවක්ම අනන්‍ය (unique) සහ වෙනස් ක්‍රියාමාර්ගයක් ගැන විය යුතුය.
9. ඉතා වැදගත්: රෝගියාට මධ්‍යස්ථ හෝ බරපතල ගැටලුවක් ඇත්නම් (max score >= 3), කිසිසේත්ම 'මෙම යහපත් සෞඛ්‍ය සම්පන්න තත්ත්වය දිගටම පවත්වා ගන්න', 'ඔබගේ මනස ඉතා සන්සුන්ව පවතින බව පෙනේ', 'ඔබගේ මානසික සෞඛ්‍යය ඉතාමත් යහපත් මට්ටමක පවතී' වැනි සුවදායී තත්ත්වයේ සිටින අයට දෙන පොදු උපදෙස් ලැයිස්තුවට ඇතුළත් නොකරන්න. ඒවා රෝගියාට නොගැලපෙන බැවින් සම්පූර්ණයෙන්ම මඟහරින්න.

පිළිතුරේ යෝජනා පමණක් ලැයිස්තුගත කරන්න (අංක යොදා). වෙනත් හැඳින්වීම් හෝ අනවශ්‍ය කතා කිසිවක් ඇතුළත් නොකරන්න."""
            
            response = gen_model.generate_content(prompt)
            # Parse the response text into a list
            suggestions = [line.strip() for line in response.text.split('\n') if line.strip()]
            
            # Clean up numbering (e.g. "1. යෝජනාව" -> "යෝජනාව") and asterisks
            cleaned_suggestions = []
            for s in suggestions:
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', s).strip()
                # Remove any single or double asterisks from start/end or anywhere in the sentence
                cleaned = cleaned.replace('**', '').replace('*', '')
                if cleaned.startswith('-'):
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
        "total_filtered": min(len(unique_texts), max_suggestions_limit),
        "suggestions": unique_texts[:max_suggestions_limit]
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

CRITICAL: Keep your responses extremely short, concise, and natural (max 2-3 sentences). Do NOT write long paragraphs or essays. Be direct, natural, and conversational, while still being empathetic.

The primary purpose of this chat is to discuss, clarify, and address the symptoms/issues the patient selected in their questionnaire profile.
- Dynamically assess the conversation: As long as there are unresolved or unclarified concerns regarding their questionnaire profile issues, continue the conversational flow by asking gentle, relevant, open-ended questions to explore them.
- Once you judge that their concerns have been sufficiently explored and they have received initial support/counseling, stop asking questions. Instead, provide a reassuring summary or tip, and gently let them know they can click the "Finish" button below to wrap up and get their final suggestions (e.g., "ඔබට අවශ්‍ය නම් පහත බොත්තමෙන් අවසන් කර යෝජනා ලබාගත හැක.").

Follow these guidelines implicitly:
1. Empathy & Validation: Acknowledge the user's input briefly.
2. CBT/Advice: Offer a reassuring perspective or a practical tip in 1 sentence.
3. Next Step: Ask one gentle question if concerns need further exploration. Otherwise, omit the question and guide them to finish the chat when they are ready.

All steps combined must fit in a single, short, and natural response of 2-3 sentences.

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
