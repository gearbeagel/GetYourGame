import re

import kagglehub
import keras
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import metrics
from textblob import TextBlob


def analyze_review_sentiment(review_text):
    if isinstance(review_text, str):
        blob = TextBlob(review_text)
        return blob.sentiment.polarity
    return 0


def parse_estimated_owners(owner_range):
    try:
        # 20000 - 50000
        lower, upper = owner_range.split(' - ')
        return (int(lower) + int(upper)) / 2
    except ValueError:
        return 0


def preprocess_data(df):
    df['combined_features'] = df['short_description'].fillna('') + " " + df['tags'].fillna('') + " " + df['genres'].fillna('')
    df['combined_features'] = df['combined_features'].apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?\'";:-]', '', x))
    df['combined_features'] = df['combined_features'].fillna('')
    df['score'] = 0
    df['review_sentiment'] = df['reviews'].apply(analyze_review_sentiment)
    df['estimated_owners_processed'] = df['estimated_owners'].apply(parse_estimated_owners)
    return df


def get_tfidf_and_scaler(df):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    additional_features = df[['review_sentiment', 'estimated_owners_processed']].to_numpy()
    combined_features = np.hstack([tfidf_matrix.toarray(), additional_features])
    scaler = StandardScaler()
    scaler.fit(combined_features)
    return tfidf, scaler


def prepare_data(df, tfidf, scaler):
    combined_features = tfidf.transform(df['combined_features']).toarray()
    additional_features = df[['review_sentiment', 'estimated_owners_processed']].values
    if additional_features.ndim == 1:
        additional_features = additional_features.reshape(-1, 1)
    assert combined_features.shape[0] == additional_features.shape[0], "Number of samples must be equal!"
    X = np.hstack([combined_features, additional_features])
    y = df['score'].to_numpy()
    X = scaler.fit_transform(X)
    return X, y


def build_keras_model(input_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01), input_dim=input_dim))
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[metrics.MeanSquaredError()])
    return model


def train_and_save_model(df, model_save_path='./data/model.keras'):
    tfidf, scaler = get_tfidf_and_scaler(df)
    X, y = prepare_data(df, tfidf, scaler)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    model = build_keras_model(input_dim=X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping])
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')


def get_keras_model(model_path='./data/model.keras'):
    model = keras.saving.load_model(model_path)
    return model


def main():
    path = kagglehub.dataset_download('artermiloff/steam-games-dataset')
    df = pd.read_csv(path + '/games_may2024_cleaned.csv')
    df = preprocess_data(df)
    train_and_save_model(df)


if __name__ == "__main__":
    main()
