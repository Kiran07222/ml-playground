import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Real-world training data (expanded dataset)
emails = [
    "Win free money now", "Claim your prize today", "Free cash reward click",
    "Click here to win big", "Limited offer act now", "Congratulations you won",
    "Verify your account urgently", "Confirm password immediately", "Update payment details",
    "Meeting at 3pm tomorrow", "Please review the report", "Lets catch up for lunch",
    "Project deadline extended", "Attached is the document", "Good morning team",
    "Thanks for your email", "Quick question about the proposal", "Can you join the call",
]

labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Create pipeline: vectorizer + classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', max_features=100)),
    ('classifier', MultinomialNB())
])

# Split data for proper evaluation
X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.3, random_state=42, stratify=labels
)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
print("=== Model Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2%}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2%}")
print(f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.2%}")

# Predict on new emails
print("\n=== Predictions ===")
new_emails = [
    "Free money reward", "See you at the meeting",
    "Urgent verify account now", "Let's discuss the agenda"
]

predictions = pipeline.predict(new_emails)
probabilities = pipeline.predict_proba(new_emails)

for email, pred, probs in zip(new_emails, predictions, probabilities):
    confidence = max(probs) * 100
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f'"{email}" -> {label} ({confidence:.1f}% confidence)')

# Save model for deployment
with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("\nModel saved as 'spam_classifier.pkl'")