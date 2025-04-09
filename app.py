# app.py
from flask import Flask, request, render_template, jsonify
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from time import sleep

app = Flask(__name__)

# Load ML model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('spam_detector.pkl', 'rb'))
except Exception as e:
    print(f"Model loading error: {e}")

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def transform(text):
    ps = PorterStemmer()
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in string.punctuation 
            and word not in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SMS Spam Guardian</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            :root {
                --primary: #4361ee;
                --danger: #f72585;
                --success: #4cc9f0;
                --dark: #212529;
                --light: #f8f9fa;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Poppins', sans-serif;
            }
            
            body {
                background: 
                    linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                    url('https://images.unsplash.com/photo-1516321318423-f06f85e504b3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
                min-height: 100vh;
                color: var(--dark);
                display: flex;
                flex-direction: column;
            }
            
            .navbar {
                background-color: #141414;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 1rem 2rem;
                backdrop-filter: blur(5px);
            }

            
            .container {
                max-width: 800px;
                margin: 2rem auto;
                padding: 0 1rem;
                flex: 1;
            }
            
            .card {
                background-color: rgba(255, 255, 255, 0.85);
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                padding: 2rem;
                margin-bottom: 2rem;
                transition: all 0.3s ease;
                backdrop-filter: blur(5px);
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.12);
            }
            
            h1 {
                color: var(--primary);
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            textarea {
                width: 100%;
                padding: 1rem;
                border: 1px solid #ddd;
                border-radius: 8px;
                resize: none;
                min-height: 150px;
                font-size: 1rem;
                transition: border 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: var(--primary);
            }
            
            .btn {
                background-color: var(--primary);
                color: white;
                border: none;
                padding: 0.8rem 2rem;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 500;
                transition: all 0.3s;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            
            .btn:hover {
                background-color: #3a56d4;
                transform: translateY(-2px);
            }
            
            .btn i {
                margin-right: 8px;
            }
            
            .result-container {
                display: none;
                animation: fadeIn 0.5s ease;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .result-box {
                padding: 1.5rem;
                border-radius: 8px;
                margin-top: 1.5rem;
                text-align: center;
            }
            
            .spam {
                background-color: rgba(247, 37, 133, 0.1);
                border-left: 4px solid var(--danger);
            }
            
            .ham {
                background-color: rgba(76, 201, 240, 0.1);
                border-left: 4px solid var(--success);
            }
            
            .result-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .spam .result-title {
                color: var(--danger);
            }
            
            .ham .result-title {
                color: var(--success);
            }
            
            .original-message {
                margin-top: 1rem;
                padding: 1rem;
                background-color: rgba(0,0,0,0.03);
                border-radius: 6px;
                font-size: 0.9rem;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 1rem 0;
            }
            
            .spinner {
                border: 4px solid rgba(0,0,0,0.1);
                border-radius: 50%;
                border-top: 4px solid var(--primary);
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            footer {
                text-align: center;
                padding: 1.5rem;
                color: #6c757d;
                font-size: 0.9rem;
                background-color: rgba(255,255,255,0.7);
                backdrop-filter: blur(5px);
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 0 1rem;
                }
                
                .card {
                    padding: 1.5rem;
                }
            }
            
            @media (prefers-color-scheme: dark) {
                body {
                    background: 
                        linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                        url('https://images.unsplash.com/photo-1516321318423-f06f85e504b3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80');
                    color: #f0f0f0;
                }
                .card {
                    background-color: rgba(30, 30, 30, 0.85);
                    color: #f0f0f0;
                }
                textarea {
                    background-color: rgba(50, 50, 50, 0.7);
                    color: white;
                }
                footer {
                    background-color: rgba(0,0,0,0.7);
                    color: #aaa;
                }
            }
        </style>
    </head>
    <body>
        <nav class="navbar">
            <div class="navbar-brand">
                <h2 style="color: var(--primary);"><i class="fas fa-shield-alt"></i> SMS Spam Guardian</h2>
            </div>
        </nav>
        
        <div class="container">
            <div class="card">
                <h1><i class="fas fa-sms"></i> Check Your Message</h1>
                
                <div class="form-group">
                    <textarea id="messageInput" placeholder="Paste your SMS message here..."></textarea>
                </div>
                
                <button id="checkBtn" class="btn">
                    <i class="fas fa-search"></i> Check for Spam
                </button>
                
                <div id="loading" class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing your message...</p>
                </div>
                
                <div id="resultContainer" class="result-container">
                    <div id="resultBox" class="result-box">
                        <div class="result-title">
                            <i id="resultIcon" class="fas"></i>
                            <span id="resultText"></span>
                        </div>
                        <div class="original-message">
                            <strong>Original message:</strong>
                            <p id="originalMessage"></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Powered by Machine Learning & Natural Language Processing</p>
        </footer>
        
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            $(document).ready(function() {
                $('#checkBtn').click(function() {
                    const message = $('#messageInput').val().trim();
                    
                    if (!message) {
                        alert('Please enter a message to check');
                        return;
                    }
                    
                    $('#loading').show();
                    $('#resultContainer').hide();
                    
                    $.ajax({
                        type: 'POST',
                        url: '/predict',
                        data: { message: message },
                        success: function(response) {
                            $('#loading').hide();
                            
                            const resultBox = $('#resultBox');
                            const resultText = $('#resultText');
                            const resultIcon = $('#resultIcon');
                            
                            if (response.error) {
                                resultBox.removeClass('spam ham').addClass('spam');
                                resultText.text('Error: ' + response.error);
                                resultIcon.removeClass().addClass('fas fa-exclamation-circle');
                            } else {
                                if (response.prediction === 'spam') {
                                    resultBox.removeClass('ham').addClass('spam');
                                    resultText.text('This message is SPAM!');
                                    resultIcon.removeClass().addClass('fas fa-exclamation-triangle');
                                } else {
                                    resultBox.removeClass('spam').addClass('ham');
                                    resultText.text('This message is HAM (safe)');
                                    resultIcon.removeClass().addClass('fas fa-check-circle');
                                }
                            }
                            
                            $('#originalMessage').text(response.message);
                            $('#resultContainer').show();
                        },
                        error: function() {
                            $('#loading').hide();
                            alert('An error occurred. Please try again.');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.form.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        processed_text = transform(message)
        vector = tfidf.transform([processed_text])
        prediction = model.predict(vector)[0]
        
        return jsonify({
            'prediction': 'spam' if prediction == 1 else 'ham',
            'message': message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)