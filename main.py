import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1. CSV
print('Atidaromas CSV')
df = pd.read_csv('dataset/medium_articles.csv')
texts = df['text'].dropna()
print(f'Skirtingu tekstu kiekis - {len(texts)}')

# DEBUG 1000 TEKSTU DEL GREICIO
texts = texts.sample(10000, random_state=42)

# 2. TF-IDF
print('TF-IDF vektorizacija')
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)
print(f'TF-IDF baigtas - {X.shape}')

# 3. K-MEANS
print('K-Means clusterization')
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df.loc[texts.index, 'tema_id'] = kmeans.fit_predict(X)
print('K-Means baigtas')

# 4. RAKTAZODZIAI
print('Isgaunami raktazodziai')
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

temos = {}
for i in range(5):
    top_words = [terms[ind] for ind in order_centroids[i, :10]]
    temos[i] = top_words
    print(f'Temai {i} raktazodziai paruosti')

# 5. TEMU GENERAVIMAS
def generuoti_temos_pavadinima(raktazodziai):
    filtruoti = [
        w for w in raktazodziai
        if len(w) > 4 and not w.isdigit()
    ]

    bigramos = [w for w in filtruoti if ' ' in w]
    if len(bigramos) >= 2:
        pasirinkti = bigramos[:2]
    else:
        pasirinkti = filtruoti[:2]

    return ' / '.join(w.title() for w in pasirinkti)

print('Generuojami temu pavazdinimai')
temu_pavadinimai = {}
for tema_id, raktazodziai in temos.items():
    pavad = generuoti_temos_pavadinima(raktazodziai)
    temu_pavadinimai[tema_id] = pavad
    print(f'Tema {tema_id} -> "{pavad}"')

print('done')

print('\n=== TEMOS ===')
for tema_id in range(5):
    print(f'Tema {tema_id}: {temu_pavadinimai[tema_id]}')
    print(f'Raktazodziai: {", ".join(temos[tema_id])}\n')
