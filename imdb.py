# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 2. LOAD DATASET
df = pd.read_csv(
    "IMDB_Dataset.csv",
    encoding_errors='ignore',
    on_bad_lines='skip',
    engine='python'
)

print("Dataset Loaded Successfully")
print(df.head())
print(df.columns)

# Remove missing values
df = df.dropna()

print(df['sentiment'].unique())
print(df['sentiment'].isnull().sum())

print(df.isnull().sum())


# 3. FIX COLUMN NAMES
df.columns = ['review', 'sentiment']



# 4. CLEAN TEXT
def clean_text(text):
    text = str(text)
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text


df['review'] = df['review'].apply(clean_text)


# 5. CONVERT LABELS
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

print("\nSentiment Count:")
print(df['sentiment'].value_counts())


# 6. TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['review']).toarray()
y = df['sentiment']


# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 8. BUILD DEEP NEURAL NETWORK
model = Sequential()

# Hidden Layer 1
model.add(Dense(
    128,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

# Hidden Layer 2
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display Model Summary
model.summary()



# 9. TRAIN MODEL
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)


# 10. PLOT ACCURACY GRAPH
plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()

plt.show()


# 11. FINAL EVALUATION
loss, accuracy = model.evaluate(X_test, y_test)

print("\nFinal Accuracy:", accuracy)