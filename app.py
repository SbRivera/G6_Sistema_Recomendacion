from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

anime_data = pd.read_csv('anime.csv', delimiter='\t')

tfidf = TfidfVectorizer(stop_words='english')
anime_data['genres'] = anime_data['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(anime_data['genres'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_anime(title, num_recommendations=5):
    idx = anime_data[anime_data['title'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations+1]
    anime_indices = [i[0] for i in sim_scores]

    return anime_data[['title', 'synopsis', 'num_episodes', 'main_pic']].iloc[anime_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    anime_name = request.form['anime_name']  # Obtener el nombre del anime del formulario
    recommendations = recommend_anime(anime_name)
    return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
