import os
import torch
import requests
from io import BytesIO
from flask import Flask, render_template, request, session, redirect, url_for
from transformers import T5Tokenizer, T5ForConditionalGeneration
from werkzeug.utils import secure_filename
import PyPDF2

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hugging Face model repository details
HF_REPO = "CodeSujal/Quiz_Generator"  # Your Hugging Face repo
MODEL_FILENAME = "fine_tuned_t5_mcq_model.pth"

# Load tokenizer from local directory
t5_tokenizer = T5Tokenizer.from_pretrained("fine_tuned_t5_mcq_tokenizer")

# Fetch and load the model from Hugging Face dynamically
model_url = f"https://huggingface.co/{HF_REPO}/resolve/main/{MODEL_FILENAME}"
print("⏳ Downloading model from Hugging Face...")

t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Ensure correct base model
state_dict = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cpu'), progress=True)
t5_model.load_state_dict(state_dict)
t5_model.eval()

print("✅ Model loaded successfully!")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join(page.extract_text().strip() for page in reader.pages if page.extract_text())
        return text if text else None
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file: {e}")

# Function to generate unique MCQs
def generate_unique_mcqs(context, num_questions):
    mcqs = []
    retries = 0

    try:
        inputs = t5_tokenizer(context, return_tensors="pt", max_length=512, truncation=True)

        while len(mcqs) < num_questions and retries < num_questions * 3:
            retries += 1
            with torch.no_grad():
                outputs = t5_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=200,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=1.0
                )

            for output in outputs:
                mcq = t5_tokenizer.decode(output, skip_special_tokens=True)
                parsed_mcq = parse_and_format_mcq_with_options(mcq)

                if parsed_mcq and parsed_mcq not in mcqs:
                    mcqs.append(parsed_mcq)

            if retries >= num_questions * 3:
                break  # Stop excessive retries

    except Exception as e:
        print(f"Error generating MCQ: {e}")
        return None

    return mcqs if len(mcqs) >= num_questions else None

# Function to parse and format MCQ with 4 options (1 correct)
def parse_and_format_mcq_with_options(mcq):
    try:
        parts = mcq.split("Correct:")
        if len(parts) < 2:
            return None  

        question_part = parts[0].strip()
        correct_answer = parts[1].strip()

        options = []
        if "Options:" in question_part:
            options_section = question_part.split("Options:")[1]
            options = [opt.strip() for opt in options_section.split(".") if opt.strip()]

        if len(options) != 4 or not correct_answer:
            return None  

        correct_option = next((opt for opt in options if correct_answer in opt), correct_answer)

        return {
            "question": question_part.split('Options:')[0].strip(),
            "options": options,
            "correct_answer": correct_option
        }
    except Exception:
        return None

# Define routes for the app
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        try:
            file = request.files.get("pdf_file")
            num_questions = request.form.get("num_questions", type=int, default=5)

            if not file or not file.filename.endswith(".pdf"):
                error = "Please upload a valid PDF file."
            else:
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(pdf_path)

                input_text = extract_text_from_pdf(pdf_path)
                if not input_text:
                    error = "PDF file is empty or could not be read."
                else:
                    quizzes = generate_unique_mcqs(input_text, num_questions)
                    if quizzes is None:
                        error = "Failed to generate enough unique MCQs."
                    else:
                        session['quizzes'] = quizzes
                        session['current_question'] = 0
                        session['score'] = 0
                        session['attempted'] = 0
                        session['skipped'] = 0
                        return redirect(url_for('quiz'))
        except Exception as e:
            error = str(e)

    return render_template("index.html", error=error)

@app.route('/quiz')
def quiz():
    current_question = session.get('current_question', 0)
    quizzes = session.get('quizzes', [])
    feedback = session.pop('feedback', None)

    if current_question >= len(quizzes):
        return redirect(url_for('game_over'))

    return render_template(
        'quiz.html',
        question=quizzes[current_question],
        question_index=current_question,
        score=session.get('score', 0),
        attempted=session.get('attempted', 0),
        skipped=session.get('skipped', 0),
        feedback=feedback
    )

@app.route('/game_over')
def game_over():
    return render_template(
        'game_over.html',
        score=session.get('score', 0),
        total_questions=len(session.get('quizzes', [])),
        attempted=session.get('attempted', 0),
        skipped=session.get('skipped', 0)
    )

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
