# Semantic-Sentiment-Analysis
🎭 IMDB Sentiment Analysis with Logistic Regression
This project builds a sentiment analysis model using machine learning and NLP techniques. It classifies IMDB movie reviews as positive or negative. A custom predict_sentiment() function is also provided to test unseen text.

📁 Dataset
Source: IMDB Large Movie Review Dataset

Size: 50,000 labeled reviews (25k positive, 25k negative)

📌 Features
Text preprocessing: HTML tag removal, lowercasing, punctuation removal, tokenization, stop word removal, lemmatization.

Vectorization using TF-IDF

Sentiment classification using Logistic Regression

Evaluation on custom test cases

Minimal dependencies, easily extensible

🧠 Workflow Overview
1. Install Dependencies
python
Copy
Edit
import nltk
nltk.download('stopwords')
nltk.download('punkt')
spacy.cli.download("en_core_web_sm")
2. Preprocess Data
python
Copy
Edit
def preprocess_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove punctuation and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatize
    doc = nlp(' '.join(tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return ' '.join(lemmatized_tokens)
3. Prepare the Dataset
python
Copy
Edit
df = pd.read_csv('IMDB Dataset.csv')
df['cleaned_review'] = df['review'].apply(preprocess_text)
df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})
4. Split & Vectorize
python
Copy
Edit
X = df['cleaned_review']
y = df['sentiment_numeric']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
5. Train the Model
python
Copy
Edit
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
model.fit(X_train_tfidf, y_train)
6. Predict Sentiment
python
Copy
Edit
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)
    return 'Positive' if prediction[0] == 1 else 'Negative'
7. Sample Test Cases
python
Copy
Edit
review_1 = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
review_2 = "I was so bored throughout the entire film. It was a complete waste of time and money."
review_3 = "The film was okay, not great but not terrible either. Some parts were good."

print(predict_sentiment(review_1))  # Expected: Positive
print(predict_sentiment(review_2))  # Expected: Negative
print(predict_sentiment(review_3))  # Depends, likely Neutral → Negative
✅ Requirements
Python 3.7+

pandas

nltk

spacy

scikit-learn

🔬 Future Improvements
Add support for neutral class using 3-class classification

Use pre-trained transformers (BERT)

Visualize sentiment distributions

Deploy as a web app (Flask or Streamlit)

🤝 Contributing
Pull requests welcome! For major changes, please open an issue first to discuss what you would like to change.

© License
This project is open-source and free to use for educational purposes.
