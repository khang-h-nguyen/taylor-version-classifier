
# Taylor Swift Tracks Classification Project

This project focuses on classifying Taylor Swift's tracks based on whether they belong to her past "stolen" albums or her re-released "Taylor's Version" albums that were recorded recently. The classification is done using logistic regression with PyTorch.

## Installation

1. Install the necessary dependencies:
   - [Python](https://www.python.org/downloads/)
   - [Spotipy](https://spotipy.readthedocs.io/en/2.19.0/) (Spotify API wrapper)
   - [PyTorch](https://pytorch.org/get-started/locally/) (for deep learning models)
   - [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) (for data manipulation)
   
2. Clone the repository to your local machine:
   ```
   git clone https://github.com/khang-h-nguyen/tsftv.git
   ```
   
3. Set up your Spotify API credentials:
   - Register your application on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
   - Obtain your client ID and client secret.
   - Replace the placeholders with your client ID and client secret in the code.

## Usage

1. Run the `import.py` script to import Taylor Swift's tracks from Spotify and save them to a CSV file.
2. Run the `classify.py` script to preprocess the data, train the logistic regression model, and evaluate its performance.

## Data

The data consists of various features extracted from Taylor Swift's tracks, including danceability, energy, loudness, and more. These features are obtained using the Spotify API.

## Models

Logistic regression is used as the classification model in this project. PyTorch is employed to implement the logistic regression model.

  
You can follow these instructions to replicate the project on your own machine and experiment with the code. Happy coding Swifties!
**Disclaimer:** This project is intended for educational purposes only. The classification of Taylor Swift's tracks is based on fictional scenarios and does not represent any real-world situation or endorsement by the artist. All rights to the music and related materials belong to their respective owners.

