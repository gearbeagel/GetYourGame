import logging
import re
from urllib.parse import urlencode

import kagglehub
import keras
import numpy as np
import pandas as pd
import requests
from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from textblob import TextBlob
from unidecode import unidecode

from data.model import get_tfidf_and_scaler
from .models import CustomUser

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.ERROR
)
logger = logging.getLogger(__name__)

STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"


def home(request):
    return render(request, 'home.html')


def steam_login(request):
    params = {
        'openid.ns': 'http://specs.openid.net/auth/2.0',
        'openid.mode': 'checkid_setup',
        'openid.return_to': request.build_absolute_uri('/steam/callback/'),
        'openid.realm': request.build_absolute_uri('/'),
        'openid.identity': 'http://specs.openid.net/auth/2.0/identifier_select',
        'openid.claimed_id': 'http://specs.openid.net/auth/2.0/identifier_select'
    }
    steam_url = f"{STEAM_OPENID_URL}?{urlencode(params)}"
    return redirect(steam_url)


def steam_logout(request):
    logout(request)
    return redirect('/')


def get_steam_username(steam_id):
    api_key = settings.STEAM_API_KEY
    url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {
        'key': api_key,
        'steamids': steam_id,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'players' in data['response']:
            player_data = data['response']['players'][0]
            username = player_data.get('personaname', 'No username found')
            return username
    return None


def steam_callback(request):
    openid_params = request.GET
    if 'openid.identity' in openid_params:
        steam_id = openid_params['openid.identity'].split('/')[-1]
        username = get_steam_username(steam_id)
        user, created = CustomUser.objects.get_or_create(steam_id=steam_id)
        if created and username:
            user.username = username
            user.save()
        backend = 'django.contrib.auth.backends.ModelBackend'
        login(request, user, backend=backend)
        return redirect('get_user_games')
    return render(request, 'error.html', {'error': 'Steam login failed.'})


@login_required
def get_user_games(request):
    steam_id = request.user.steam_id
    response = requests.get(
        "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
        params={
            'key': settings.STEAM_API_KEY,
            'steamid': steam_id,
            'include_appinfo': True,
            'format': 'json',
        }
    )
    if response.status_code == 200:
        games_data = response.json().get('response', {}).get('games', [])
        game_details = []
        for game in games_data:
            appid = game['appid']
            store_response = requests.get(f"https://store.steampowered.com/api/appdetails", params={"appids": appid})
            if store_response.status_code == 200:
                store_data = store_response.json().get(str(appid), {}).get("data", {})
                short_description = store_data.get('short_description', 'No description available')
                cover_url = store_data.get('header_image',
                                           f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg")
            else:
                short_description = 'No description available'
                cover_url = f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/header.jpg"
            game_details.append(
                {'name': game['name'], 'short_description': short_description, 'cover_url': cover_url, 'appid': appid})
        return render(request, 'your_games.html', {'games': game_details})
    return render(request, 'error.html', {'error': 'Unable to retrieve games'})


path = kagglehub.dataset_download('artermiloff/steam-games-dataset')
df = pd.read_csv(path + '/games_may2024_cleaned.csv', encoding='utf-8-sig')


def analyze_review_sentiment(review_text):
    if isinstance(review_text, str):
        blob = TextBlob(review_text)
        return blob.sentiment.polarity
    return 0


def parse_estimated_owners(owner_range):
    try:
        lower, upper = owner_range.split(' - ')
        return (int(lower) + int(upper)) / 2
    except ValueError:
        return 0


def clean_text(text):
    if isinstance(text, str):
        text = unidecode(re.sub(r'[^a-zA-Z0-9\s.,!?\'";:-]', '', text))
    return text


def preprocess_data(df):
    df['short_description'] = df['short_description'].apply(clean_text)
    df['tags'] = df['tags'].apply(clean_text)
    df['genres'] = df['genres'].apply(clean_text)
    df['reviews'] = df['reviews'].apply(clean_text)
    df['combined_features'] = df['short_description'].fillna('') + " " + df['tags'].fillna('') + " " + df[
        'genres'].fillna('')
    df['combined_features'] = df['combined_features'].apply(clean_text)
    df['score'] = 0
    df['review_sentiment'] = df['reviews'].apply(analyze_review_sentiment)
    df['estimated_owners_processed'] = df['estimated_owners'].apply(parse_estimated_owners)
    return df


df = preprocess_data(df)


def get_keras_model(model_path='./data/model.keras'):
    model = keras.saving.load_model(model_path)
    return model


def get_user_games_data(steam_id):
    response = requests.get(
        "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/",
        params={
            'key': settings.STEAM_API_KEY,
            'steamid': steam_id,
            'include_appinfo': True,
            'format': 'json',
        }
    )
    if response.status_code == 200:
        user_games_data = response.json().get('response', {}).get('games', [])
        return [{'name': game['name'], 'appid': game['appid']} for game in user_games_data]
    return []


@login_required
def recommend_view(request):
    steam_id = request.user.steam_id
    user_games = get_user_games_data(steam_id)

    if not user_games:
        return render(request, 'error.html', {'error': 'Unable to retrieve games for recommendations'})

    recommendations = []

    if request.method == 'GET' and 'generate_recommendations' in request.GET:
        model = get_keras_model()
        tfidf, scaler = get_tfidf_and_scaler(df)

        owned_game_names = [game['name'] for game in user_games]
        filtered_df = df[~df['name'].isin(owned_game_names)]

        combined_features_tfidf = tfidf.transform(filtered_df['combined_features'])

        additional_features = filtered_df[['review_sentiment', 'estimated_owners_processed']].to_numpy()

        game_features = np.hstack([
            combined_features_tfidf.toarray(),
            additional_features
        ])

        game_features_scaled = scaler.transform(game_features)

        predicted_scores = model.predict(game_features_scaled, verbose=0)

        filtered_df['predicted_score'] = predicted_scores

        recommendations = (
            filtered_df[
                ['name', 'predicted_score', 'tags', 'genres', 'AppID', 'reviews', 'short_description', 'header_image']]
            .sort_values('predicted_score', ascending=False)
            .head(200)
            .sample(5)
            .to_dict(orient='records')
        )

    return render(request, 'recommendations.html',
                  {'user_games': user_games, 'content_recommendations': recommendations})
