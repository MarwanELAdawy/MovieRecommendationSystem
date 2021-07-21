import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# We use pandas to read the CSV dataset and load it to df (DataFrame) variable.
df = pd.read_csv("movie_dataset.csv")


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]


# we need to get the feature that we need to use to find a similarity score. For now, I’m going to use keywords, cast, genres, and director

features = ['keywords', 'cast', 'genres', 'director']

#Let’s now combine all the features to the dataframe
for feature in features:
    df[feature] = df[feature].fillna('')


#And a function to combine them into one:
def combine_features(row):
    try:
        return row['keywords'] + " " + row['cast'] + " " + row["genres"] + "   " + row["director"]
    except:
        print("Error:", row)


df["combined_features"] = df.apply(combine_features, axis=1)

print("Combined Features:", df["combined_features"].head())

#let’s initialise count vectoriser to turn them into vectors which we can work with
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#Getting the similiarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

#Now we take the input from the user. That is the liked movie by the user
movie_user_likes = "Spectre"


#get the index of the movie from the name of the movie
movie_index = get_index_from_title(movie_user_likes)


#Get the list of similar movies (indices of similar movies)
similar_movies = list(enumerate(cosine_sim[movie_index]))


#we need the highest similar movie on the top so we sort it to descending order
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)


#we’ll print the most matched 25 movies to the console(output)
i = 0
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i = i + 1
    if i > 25:
        break
