import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'backend', '.env'))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in backend/.env")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

themes = [
    "Depression and extreme sadness (විෂාදය සහ දැඩි දුක)",
    "Generalized Anxiety and Panic Attacks (කාංසාව සහ කලබලවීම)",
    "Workplace Stress and Burnout (රැකියා ආතතිය)",
    "Relationship and Family Issues (පවුල් සහ සබඳතා ගැටලු)",
    "Sleep Disorders and Insomnia (නින්ද නොයාම)",
    "Trauma and PTSD (කම්පනය සහ අතීත සිදුවීම්)",
    "OCD and intrusive thoughts (අනවශ්‍ය සිතුවිලි සහ OCD)",
    "Anger Management (කෝපය පාලනය කිරීම)",
    "Social Anxiety and Shyness (සමාජ භීතිකාව)",
    "Low Self-Esteem and Confidence Issues (ආත්ම විශ්වාසය අඩුවීම)",
    "Addiction and Substance Abuse (ඇබ්බැහිවීම්)",
    "Eating Disorders (ආහාර ගැනීමේ ගැටලු)",
    "Grief and Loss (වියෝ දුක)",
    "Academic Pressure and Exam Stress (අධ්‍යාපන පීඩනය)",
    "Financial Stress (මූල්‍යමය ආතතිය)"
]

def get_gemini_suggestions(theme):
    prompt = f"""You are generating a dataset for a psychological AI.
Topic: {theme}
Generate 50 distinct, highly practical, and empathetic clinical counseling suggestions in Sinhala for someone experiencing this problem.
Format strictly as a list, each starting with a dash (-). Do not include any other text or English words.
Make sure the Sinhala is natural and grammatically correct."""

    print(f"Generating 50 suggestions for: {theme}...")
    try:
        response = model.generate_content(prompt)
        suggestions = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                cleaned = line[1:].strip()
                if cleaned:
                    suggestions.append(cleaned)
        return suggestions
    except Exception as e:
        print(f"Failed for {theme}: {e}")
        return []

def main():
    output_file = "global_suggestions_dataset.txt"
    dataset_path = os.path.join("chatbot-web", "src", "dataset.json")
    
    formatted_data = []

    # 1. Extract existing 634 suggestions from JSON
    print("Extracting JSON dataset...")
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for q in data.get("questionnaire", []):
                question_text = q.get("question", "")
                for opt in q.get("options", []):
                    for sug in opt.get("suggestions", []):
                        pair = f"User: මට පහත ගැටලු ඇත: {question_text}\nBot: {sug}\n"
                        formatted_data.append(pair)
    else:
        print(f"Warning: {dataset_path} not found.")

    print(f"Extracted {len(formatted_data)} pairs from JSON.")

    # 2. Generate new suggestions via Gemini
    for theme in themes:
        sugs = get_gemini_suggestions(theme)
        # Create pairs for these
        theme_sinhala = theme.split('(')[1].replace(')', '')
        for sug in sugs:
            pair = f"User: මට පහත ගැටලු ඇත: {theme_sinhala}\nBot: {sug}\n"
            formatted_data.append(pair)
        time.sleep(2) # Rate limit protection

    print(f"Total unique pairs generated: {len(formatted_data)}")

    # 3. Duplicate data to ensure SLM learns properly (Bulk up for GPT-2 training)
    # 30M models need a decent amount of iterations. We will duplicate the dataset 15 times to create a larger file.
    multiplied_data = formatted_data * 15

    print(f"Writing {len(multiplied_data)} pairs to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(multiplied_data))
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
