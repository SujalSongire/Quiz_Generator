# 📘 Quiz Generator from PDFs using T5 Model

This project is a Flask-based web application that generates Multiple Choice Questions (MCQs) from PDF documents using a fine-tuned T5 model. Users can upload a PDF, extract its text, and generate quiz questions automatically.

## ✨ Features
- 📂 Upload a PDF file to extract text.
- 🧠 Generate unique MCQs from extracted text using a fine-tuned T5 model.
- 🎯 Interactive quiz interface with scoring and navigation.
- 📊 Track the number of attempted, skipped, and correct answers.
- 🔄 Session management to maintain quiz progress.

## 🛠️ Tech Stack
- **Backend:** Flask (Python)
- **Machine Learning Model:** Hugging Face Transformers (T5 fine-tuned model)
- **Frontend:** HTML, CSS, Jinja Templates
- **Storage:** Local file system for uploads

## 🚀 Setup Instructions

### ✅ Prerequisites
- Python 3.7+
- pip package manager

### 📥 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/quiz-generator.git
   cd quiz-generator
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional):**
   ```sh
   export SECRET_KEY='your_secret_key'
   ```

5. **Run the Flask app:**
   ```sh
   python app.py
   ```
   Then, open `http://127.0.0.1:5000/` in your browser.

## 🎮 Usage
1. Navigate to the homepage and upload a PDF file.
2. Choose the number of MCQs to generate.
3. Start the quiz and select your answers.
4. View your score at the end of the quiz.

## 📂 Project Structure
```
quiz-generator/
├── fine_tuned_t5_mcq_tokenizer/   # Tokenizer directory
├── static/                        # Static files (CSS, JS, Images)
├── templates/                     # HTML templates
│   ├── index.html                 # Upload page
│   ├── quiz.html                   # Quiz interface
│   ├── game_over.html              # Result page
├── uploads/                       # Directory for storing uploaded PDFs
├── app.py                         # Flask application
├── requirements.txt               # List of dependencies
├── procfile                       # Heroku deployment config
└── README.md                      # Project documentation
```

## 🧠 Model Details
- This project uses a **fine-tuned T5 model** hosted on Hugging Face.
- The model is loaded directly from Hugging Face and used to generate MCQs from input text.

## 🔍 Troubleshooting
- ⚠️ If the model does not load, ensure you have a stable internet connection.
- 🛠️ Check if PyTorch and Transformers libraries are installed properly.
- 🔑 If `SECRET_KEY` error appears, set an environment variable for it.

## 🤝 Contributing
Feel free to fork this repository, raise issues, and submit pull requests to improve this project.

## 📜 License
This project is open-source and available under the MIT License.

