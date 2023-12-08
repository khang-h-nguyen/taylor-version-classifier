import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Set up your Spotify API credentials
client_id = '6caf19aef7714b30b0e4f1b4e7e3efd8'
client_secret = 'e7ef2b60c99d40c4ba6cd7f0273de982'
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Function to get track features from Spotify API
def get_track_features(track_id):
    track_info = sp.track(track_id)
    features = sp.audio_features(track_id)[0]
    
    return {
        'track_name': track_info['name'],
        'track_id': track_id,
        'album_name': track_info['album']['name'],
        'album_id': track_info['album']['id'],
        'artist_name': track_info['artists'][0]['name'],
        'artist_id': track_info['artists'][0]['id'],
        'release_date': track_info['album']['release_date'],
        'danceability': features['danceability'],
        'energy': features['energy'],
        'key': features['key'],
        'loudness': features['loudness'],
        'mode': features['mode'],
        'speechiness': features['speechiness'],
        'acousticness': features['acousticness'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'valence': features['valence'],
        'tempo': features['tempo'],
        'duration_ms': features['duration_ms'],
        'time_signature': features['time_signature']
    }

# Function to get tracks from an album
def get_album_tracks(album_id, ftv = False):
    tracks = sp.album_tracks(album_id)
    track_ids = []
    if ftv:
        for track in tracks['items']:
            if 'From The Vault' in track['name']:
                track_ids.append(track['id'])
    else:
        track_ids.extend([track['id'] for track in tracks['items']])
    return track_ids

# Function to get tracks containing 'From The Vault' in the name
def get_from_the_vault_tracks(artist_id):
    all_tracks = []

    albums = sp.artist_albums(artist_id, album_type='album')
    for album in albums['items']:
        album_id = album['id']
        album_name = album['name']
        if 'Taylor\'s Version' in album_name:
            track_ids = get_album_tracks(album_id, ftv = True)
            all_tracks.extend(track_ids)
    return all_tracks

# Function to get tracks from Midnights (The Til Dawn Edition)
def get_midnight_deluxe_tracks(artist_id):
    all_tracks = []
    albums = sp.artist_albums(artist_id, album_type='album')
    for album in albums['items']:
        album_id = album['id']
        album_name = album['name']
        if 'Midnights (The Til Dawn Edition)' == album_name:
            track_ids = get_album_tracks(album_id)
            all_tracks.extend(track_ids)
    return all_tracks

# Set Taylor Swift's artist ID
taylor_swift_artist_id = '06HL4z0CvFAxyc27GXpf02'

# Get (From The Vault) tracks and Midnights (The Til Dawn Edition)'s tracks
from_the_vault_tracks = get_from_the_vault_tracks(taylor_swift_artist_id)
midnight_deluxe_tracks = get_midnight_deluxe_tracks(taylor_swift_artist_id)

# Combine the track lists
all_tracks = from_the_vault_tracks + midnight_deluxe_tracks

# Collect data for each track
data = []
for track_id in all_tracks:
    features = get_track_features(track_id)
    data.append(features)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('taylor_swift_tracks.csv', index=False)
