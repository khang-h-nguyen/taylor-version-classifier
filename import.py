import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Set up your Spotify API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'
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
def get_album_tracks(album_id):
    tracks = sp.album_tracks(album_id)
    return [track['id'] for track in tracks['items']]

# Function to get tracks from Taylor Swift's catalogue
def get_ts_tracks(artist_id):
    all_tracks = []

    albums = sp.artist_albums(artist_id, album_type='album')
    for album in albums['items']:
        album_id = album['id']
        track_ids = get_album_tracks(album_id)
        all_tracks.extend(track_ids)
    return all_tracks

# Set Taylor Swift's artist ID
taylor_swift_artist_id = '06HL4z0CvFAxyc27GXpf02'

# Get all her tracks
all_tracks = get_ts_tracks(taylor_swift_artist_id)

# Collect data for each track
data = []
for track_id in all_tracks:
    features = get_track_features(track_id)
    data.append(features)

# Create a DataFrame
df = pd.DataFrame(data)
df['track_name'] = df['track_name'].str.replace(r'Taylor.*?Version', 'Taylor\'s Version', regex=True)

# Save the DataFrame to a CSV file
df.to_csv('taylor_swift_tracks.csv', index=False)
