from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Load the saved model and tokenizer
model = load_model('cnn_sent_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define maxlen (the same as during training)
maxlen = 100

# Initialize Flask app
app = Flask(__name__)


# Clean input text (you can modify this with your cleaning function)
# Download NLTK resources (only the first time)
nltk.download('stopwords')

# Initialize the stemmer and stopwords list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_review_with_stopwords_and_stemming(review):
    # Decode HTML entities (like <br /> to actual line breaks)
    review = html.unescape(review)

    # Remove HTML tags (e.g., <br />)
    review = re.sub(r'<.*?>', '', review)

    # Convert to lowercase
    review = review.lower()

    # Remove non-alphabetic characters and digits
    review = re.sub(r'[^a-z\s]', '', review)

    # Tokenization: Split the review into words
    words = review.split()

    # Remove stopwords and apply stemming
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Rejoin the words back into a cleaned string
    cleaned_review = ' '.join(cleaned_words)

    return cleaned_review

# Prediction function
def predict_sentiment(text):
    # Clean the text
    text = clean_review_with_stopwords_and_stemming(text)

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=maxlen)

    # Make the prediction
    prediction = model.predict(padded_sequence)

    # Return the result
    if prediction > 0.5:
        return "Positive"
    else:
        return "Negative"


# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle the form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        text = request.form['review_text']

        # Get the sentiment prediction
        sentiment = predict_sentiment(text)

        return render_template('index.html', sentiment=sentiment)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)