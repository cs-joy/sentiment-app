import pickle
import re
from django.shortcuts import render
from django.http import JsonResponse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


class SentimentAnalyzer:
    def __init__(self):
        # Load the model and vectorizer
        with open('analyzer/ml_model/sentiment_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('analyzer/ml_model/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def clean_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)
    
    def predict(self, text):
        cleaned_text = self.clean_text(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        return "Positive" if prediction == 1 else "Negative"

analyzer = SentimentAnalyzer()

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text:
            result = analyzer.predict(text)
            return JsonResponse({'sentiment': result})
        return JsonResponse({'error': 'No text provided'}, status=400)
    return render(request, 'analyzer/index.html')
