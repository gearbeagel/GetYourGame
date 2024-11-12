import logging
from urllib.parse import urlencode
from textblob import TextBlob
import pandas as pd
import torch
import requests
from django.conf import settings
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from .models import CustomUser


def home(request):
    return render(request, 'home.html')


STEAM_OPENID_URL = "https://steamcommunity.com/openid/login"


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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv('./data/games_may2024_cleaned.csv')


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


def preprocess_data(df):
    df['review_sentiment'] = df['reviews'].apply(analyze_review_sentiment)
    df['estimated_owners_processed'] = df['estimated_owners'].apply(parse_estimated_owners)
    return df


df = preprocess_data(df)


def recommend_games(game_name, df, n):
    game = df[df['name'].str.contains(game_name, case=False, na=False)]
    if game.empty:
        return pd.DataFrame()

    genres = game['genres'].values[0].strip("[]").replace("'", "").split(", ")
    tags = game['tags'].values[0].strip("[]").replace("'", "").split(", ")
    review_sentiment = game['review_sentiment'].values[0]
    game_owner_count = game['estimated_owners_processed'].values[0]

    genre_tensor = torch.tensor([1 if genre in genres else 0 for genre in df['genres'].unique()])
    tag_tensor = torch.tensor([1 if tag in tags else 0 for tag in df['tags'].unique()])

    mask = (df['genres'].apply(lambda x: any(genre in x for genre in genres)) |
            df['tags'].apply(lambda x: any(tag in x for tag in tags))) & (df['name'] != game_name)
    recommendations = df[mask].copy()

    if game_owner_count > 0:
        owners_tensor = torch.tensor(recommendations['estimated_owners_processed'].values)
        mask = (owners_tensor > game_owner_count * 0.8) & (owners_tensor < game_owner_count * 1.2)
        recommendations = recommendations[mask.numpy()]

    if review_sentiment > 0:
        sentiment_tensor = torch.tensor(recommendations['review_sentiment'].values)
        mask = sentiment_tensor > 0
    else:
        sentiment_tensor = torch.tensor(recommendations['review_sentiment'].values)
        mask = sentiment_tensor < 0
    recommendations = recommendations[mask.numpy()]

    recommendations = recommendations.sort_values(by='user_score', ascending=False)
    return recommendations.sample(n)[
        ['name', 'short_description', 'header_image', 'user_score', 'genres', 'tags', 'AppID', 'estimated_owners',
         'reviews']]


def generate_random_recommendations(user_games, df):
    content_recommendations = []

    for game_data in user_games:
        game_name = game_data['name']
        game_recommendations = recommend_games(game_name, df, n=1)

        if not game_recommendations.empty:
            game_recommendations['cover_url'] = game_recommendations['header_image']
            content_recommendations.append(game_recommendations)

    if content_recommendations:
        content_recommendations = pd.concat(content_recommendations).drop_duplicates(subset=['name'])
        content_recommendations = content_recommendations.sample(frac=1).reset_index(drop=True)
        content_recommendations = content_recommendations.to_dict(orient='records')
    else:
        content_recommendations = []

    return content_recommendations


@login_required
@login_required
def recommend_view(request):
    steam_id = request.user.steam_id

    user_games = get_user_games_data(steam_id)

    if not user_games:
        # logger.error("Failed to retrieve user games.")
        return render(request, 'error.html', {'error': 'Unable to retrieve games for recommendations'})

    content_recommendations = []
    if request.method == 'GET' and 'generate_recommendations' in request.GET:
        content_recommendations = generate_random_recommendations(user_games, df)

        content_recommendations = content_recommendations[:5]

        if not content_recommendations:
            # logger.error("No recommendations found.")
            return render(request, 'error.html', {'error': 'No recommendations available'})

    logger.info(f"Total unique recommendations generated: {len(content_recommendations)}")

    return render(request, 'recommendations.html', {
        'user_games': user_games,
        'content_recommendations': content_recommendations
    })
