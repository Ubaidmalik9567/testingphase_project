import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize FastAPI
app = FastAPI()

# Pydantic model for input data
class TextInput(BaseModel):
    text: str

def load_model_and_vectorizer(model_path, vectorizer_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

def text_preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    stemming = PorterStemmer()
    for i in text:
        y.append(stemming.stem(i))
    return " ".join(y)

def calculate_features(text):
    # Calculate number of characters, words, and sentences
    no_of_characters = len(text)
    words = nltk.word_tokenize(text)
    no_of_words = len(words)
    sentences = nltk.sent_tokenize(text)
    no_of_sentences = len(sentences)
    
    return no_of_characters, no_of_words, no_of_sentences

def prepare_input_features(input_text_transformed, no_of_characters, no_of_words, no_of_sentences):
    minmax = MinMaxScaler()
    text_features_df = pd.DataFrame(input_text_transformed.toarray(),
                                    columns=[f"feature_{i}" for i in range(input_text_transformed.shape[1])])
    numerical_features = pd.DataFrame([[no_of_characters, no_of_words, no_of_sentences]],
                                      columns=['no_of_characters', 'no_of_words', 'no_of_sentences'])
    numerical_features[['no_of_characters','no_of_words']] = minmax.fit_transform(numerical_features[['no_of_characters','no_of_words']])
    
    query = pd.concat([text_features_df, numerical_features], axis=1)    
    return query

def predict(text, model, vectorizer):
    preprocessed_text = text_preprocessing(text)
    no_of_characters, no_of_words, no_of_sentences = calculate_features(text)
    
    vectorized_text = vectorizer.transform([preprocessed_text])
    user_query = prepare_input_features(vectorized_text, no_of_characters, no_of_words, no_of_sentences)
    print(user_query.shape)
    prediction = model.predict_proba(user_query)
    # Access the first element to get probabilities for the input text
    
    ham_probability = prediction[0][0]
    spam_probability = prediction[0][1]
    return f"ham: {ham_probability:.2f}, spam: {spam_probability:.2f}, no_of_characters: {no_of_characters:.2f}, no_of_words: {no_of_words:.2f}, no_of_sentences: {no_of_sentences:.2f}"

# def predict(text, model, vectorizer):
#     preprocessed_text = text_preprocessing(text)
#     vectorized_text = vectorizer.transform([preprocessed_text])
#     prediction = model.predict_proba(vectorized_text)
#     # Access the first element to get probabilities for the input text
#     ham_probability = prediction[0][0]
#     spam_probability = prediction[0][1]
#     return f"ham: {ham_probability:.2f}, spam: {spam_probability:.2f}"

# HTML content for the web interface
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Classification</title>
    <script>
        async function getPrediction() {
            const text = document.getElementById("inputText").value;
            const response = await fetch("/predict/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            document.getElementById("prediction").innerText = "Prediction: " + data.prediction;
        }
    </script>
</head>
<body>
    <h1>Text Classification</h1>
    <textarea id="inputText" placeholder="Enter your text"></textarea>
    <button onclick="getPrediction()">Predict</button>
    <div id="prediction"></div>
</body>
</html>
"""

# Define API endpoint to serve HTML content
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return html_content

# Define API endpoint for prediction
@app.post("/predict/")
async def get_prediction(text_input: TextInput):
    text = text_input.text

    # Path to the trained model and vectorizer
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

    # Load the trained model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Make predictions
    prediction = predict(text, model, vectorizer)

    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
