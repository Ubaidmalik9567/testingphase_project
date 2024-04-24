'''import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

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

def predict(text, model, vectorizer):
    preprocessed_text = text_preprocessing(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return prediction

def main():
    # Path to the trained model and vectorizer
    model_path = "models/model.pkl"
    vectorizer_path = "models/vectorizer.pkl"

    # Load the trained model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Get input text from the user
    user_input = input("Enter your text: ")

    # Make predictions
    prediction = predict(user_input, model, vectorizer)

    print("Prediction:", prediction)

if __name__ == "__main__":
    main()


'''

import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Initialize FastAPI
app = FastAPI()

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

def predict(text, model, vectorizer):
    preprocessed_text = text_preprocessing(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    # Convert prediction to "ham" or "spam"
    if prediction[0] == 0:
        return "ham Email"
    elif prediction[0] == 1:
        return "spam Email"

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
async def get_prediction(request: Request):
    data = await request.json()
    text = data.get("text")

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
