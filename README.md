# train_lastfm

Train a regressor/classifier with last.mf dataset

## preprocess_lastfm_top50.py
* Spotify features header:
```
track_id,track_uri,track_name,artist,duration,release_date_estimate,us_popularity_estimate,
album_name,acousticness,beat_strength,bounciness,danceability,dyn_range_mean,energy,flatness,
instrumentalness,key,liveness,loudness,mechanism,mode,organism,speechiness,tempo,
time_signature,valence,acoustic_vector_0,acoustic_vector_1,acoustic_vector_2,
acoustic_vector_3,acoustic_vector_4,acoustic_vector_5,acoustic_vector_6,acoustic_vector_7
```
* Top 50 tags are selected as:
```
TOP50A = ['rock', 'pop', 'alternative', 'indie', 'favorites', 'female vocalists',
          'Love', 'alternative rock', 'electronic', 'beautiful', 'jazz', '00s',
          'singer-songwriter', 'metal', 'male vocalists', 'Awesome', 'american', 
          'Mellow', 'classic rock', '90s', 'soul', 'chillout', 'punk', '80s', 'chill',
          'indie rock', 'folk', 'dance', 'instrumental', 'hard rock', 'oldies',
          'seen live', 'Favorite', 'country', 'blues', 'guitar', 'cool', 'british',
          'acoustic', 'electronica', '70s', 'Favourites', 'Hip-Hop', 'experimental',
          'easy listening', 'female vocalist', 'ambient', 'punk rock', 'funk', 'hardcore']
```     
* (Load ./data/*.json) --> (Shuffle) --> (Check availability of top50 tags) --> (Reduce Unavailable items) --> (save as ./data/*.npy)


## Model
* TBA