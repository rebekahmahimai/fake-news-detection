
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasets/fake-news/master/data/fake.csv')

# Keep only text and label
df = df[['text', 'label']]
df.dropna(inplace=True)

# Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# Test on sample
sample_text = ["Breaking news! You won a lottery worth 5 crores."]
sample_vec = vectorizer.transform(sample_text)
print("Prediction:", model.predict(sample_vec)[0])
