from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Cargar datos de anime
anime_data = pd.read_csv('anime.csv', delimiter='\t')

# Crear la matriz TF-IDF para el campo 'genre'
tfidf = TfidfVectorizer(stop_words='english')
anime_data['genres'] = anime_data['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(anime_data['genres'])

# Calcular la similitud coseno
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función para obtener recomendaciones basadas en el nombre del anime
def recommend_anime(title, num_recommendations=5):
    # Obtener el índice del anime dado su título
    idx = anime_data[anime_data['title'] == title].index[0]

    # Obtener las similitudes del anime con todos los demás animes
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar los animes por puntuación de similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de los animes más similares
    sim_scores = sim_scores[1:num_recommendations+1]
    anime_indices = [i[0] for i in sim_scores]

    # Devolver un DataFrame con los títulos, sinopsis, número de episodios y la imagen principal de los animes recomendados
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
