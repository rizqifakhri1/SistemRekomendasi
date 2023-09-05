import string 
import pandas as pd
import nltk
import json
from flask import Flask, request, jsonify
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter

app = Flask(__name__)


#Import Dataset
sheet_id = '1_J0lxAzX28kF-YB7qdyFGMyUfHUYjo2swQ8UBunk-Zw'
xls = pd.ExcelFile(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx")
dataset = pd.read_excel(xls, 'csvdata')

# Menampilkan semua kolom dan baris dari dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max.columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max.colwidth', None)

# membuat fungsi untuk menghapus tanda baca
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

#untuk membuat teks menjadi huruf kecil
dataset["processed_judul"] = [entry.lower() for entry in dataset['Judul']]
dataset["processed_dosen"] = [entry.lower() for entry in dataset['Dosen']]
dataset["index"]=[i for i in range(0,dataset.shape[0])]

# membuat kolom baru yang berisi teks tanpa tanda baca
dataset["processed_dosen"] = [remove_punctuation(entry.lower()) for entry in dataset['processed_dosen']]
dataset["processed_judul"] = [remove_punctuation(entry.lower()) for entry in dataset['processed_judul']]

# Membaca data sinonim dari file JSON
with open('dataset.json') as json_file:
    synonyms = json.load(json_file)

# Membuat fungsi untuk mengganti sinonim dalam teks
def replace_synonyms(text, synonyms):
    for key, value in synonyms.items():
        for synonym in value['sinonim']:
            text = text.replace(synonym, key)
    return text

# Iterasi melalui semua baris data dalam dataset
for index, row in dataset.iterrows():
    # Mengambil teks dari hasil processed_judul
    judul = row["processed_judul"]

    # Memproses teks judul tanpa tanda baca
    processed_judul = remove_punctuation(judul.lower())

    # Mengganti sinonim dalam teks judul yang telah diproses
    modified_judul = replace_synonyms(processed_judul, synonyms)

    # Memasukkan hasil modified_judul ke dalam processed_judul pada dataset
    dataset.at[index, "processed_judul"] = modified_judul

# Buat list kosong untuk menampung semua kata yang muncul pada judul
all_words = []

# Looping setiap judul dan split kata-kata yang ada di dalamnya
for judul in dataset["processed_judul"]:
    words = judul.split()
    all_words += words

# Hitung frekuensi masing-masing kata
word_freq = Counter(all_words)

# Tampilkan kata dengan frekuensi tertinggi dalam bentuk tabel
top_words = word_freq.most_common()
df_top_words = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])
# print(df_top_words)
df_top_words

# create a copy of the subset
dataset_processed = dataset['processed_judul']

# remove stopwords and other words from the 'processed' column
stop = stopwords.words('indonesian')

# tambahkan stopword sastrawi
factory = StopWordRemoverFactory()
sastrawi_stopwords = factory.get_stop_words()
stop.extend(sastrawi_stopwords)

# seleksi secara manual kalimat yang sering muncul
remove = ['sistem','berbasis', 'pengembangan', 'metode', 'implementasi', 'teknik', 'hasil', 'analisis', 'penggunaan', 'rancang', 'bangun']
stop.extend(remove)

# apply stopword removal to the 'processed' column
dataset_processed = dataset_processed.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# add the processed column back to the original dataset
dataset['processed_judul'] = dataset_processed

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

abaikan = ['berbasis', 'perancangan', 'ungaran', 'pemalang', 'tahani', 'kejuruan', 'pelajaran']

factory = StemmerFactory()
stemmer = factory.create_stemmer()

results = []
for index, row in dataset.iterrows():
    judul = row['Judul']
    processed_judul = row['processed_judul']
    words = processed_judul.split()
    stemmed_words = []
    for word in words:
        if word not in abaikan:
            stemmed_word = stemmer.stem(word)
            if stemmed_word != word:
                results.append({'Judul': judul, 'Stemmed Word': word, 'Bentuk dasar': stemmed_word})
            stemmed_words.append(stemmed_word)
        else:
            stemmed_words.append(word)
    processed_stemmed_judul = ' '.join(stemmed_words)
    dataset.at[index, 'processed_stemmed_judul'] = processed_stemmed_judul
    
stemmed_table = pd.DataFrame(results)
stemmed_table

# Membuat objek vectorizer untuk pembobotan TF-IDF
vectorizer = TfidfVectorizer()

# Melakukan pembobotan TF-IDF pada dataset
docs_tfidf = vectorizer.fit_transform(dataset['processed_stemmed_judul'])

# Fungsi untuk mendapatkan nilai cosine similarity antara query dan dokumen
def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
    # Mengubah query menjadi vektor TF-IDF
    query_tfidf = vectorizer.transform([query])
    # Menghitung nilai cosine similarity antara query dan seluruh dokumen
    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf)
    return cosineSimilarities

# Fungsi untuk mendapatkan indeks dokumen yang mirip dengan query
def get_similar_document_index(query):
    # Menghitung nilai cosine similarity antara query dan seluruh dokumen menggunakan fungsi sebelumnya
    check = get_tf_idf_query_similarity(vectorizer, docs_tfidf, query)[0]
    res = []
    # Looping untuk memasukkan dokumen yang memiliki nilai cosine similarity di atas 0 ke dalam res
    for i in range(len(check)):
        rank = check[i]
        if rank > 0.2:
            t = {"index": i, "rank": rank}
            res.append(t)
    return res


# Fungsi untuk memproses query input
def processing_query_input(query):
    # Membaca data sinonim dari file JSON
    with open('dataset.json') as json_file:
        synonyms = json.load(json_file)

    # Hapus tanda baca dan ubah menjadi huruf kecil
    query = remove_punctuation(query.lower())

    # Ganti sinonim dalam query
    query = replace_synonyms(query, synonyms)

    # Hapus stopwords
    factory = StopWordRemoverFactory()
    sastrawi_stopwords = factory.get_stop_words()
    stop = stopwords.words('indonesian')
    stop.extend(sastrawi_stopwords)

    # Kata-kata tambahan yang ingin dihapus secara manual
    remove = ['berbasis', 'pengembangan', 'metode', 'pengaruh', 'implementasi', 'teknik', 'hasil', 'analisis', 'penggunaan', 'rancang', 'bangun']
    stop.extend(remove)

    # Proses menghapus stopwords
    query = ' '.join([word for word in query.split() if word not in stop])

    # Stemming kata dalam query
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = query.split()
    stemmed_words = []
    for word in words:
        if word not in abaikan:
            stemmed_word = stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        else:
            stemmed_words.append(word)
    query = ' '.join(stemmed_words)

    return query


import numpy as np
import pandas as pd

def get_recomendation(query):
    stemmed_query = processing_query_input(query)
    
    similar_doc = get_similar_document_index(stemmed_query)
    
    res = pd.DataFrame(columns=['Dosen', 'Judul', 'Count', 'Nilai Maksimum', 'Rata-rata', 'Rank'])
    for x in similar_doc:
        doc = dataset.loc[dataset['index'] == x['index']]
        judul = doc.Judul.values[0]
        dosen = doc.Dosen.values[0]
        processed_stemmed_judul = doc.processed_stemmed_judul.values[0]
        rank = x['rank']
        if dosen in res['Dosen'].values:
            res.loc[res['Dosen'] == dosen, 'Judul'] += f", {judul} ({rank})"
            res.loc[res['Dosen'] == dosen, 'Nilai Maksimum'] = max(res.loc[res['Dosen'] == dosen, 'Nilai Maksimum'].max(), rank)
            res.loc[res['Dosen'] == dosen, 'Rata-rata'] = res.loc[res['Dosen'] == dosen, 'Rata-rata'] + rank
        else:
            out = pd.DataFrame([{'Judul': f"{judul} ({rank})", 'Dosen': dosen, 'Nilai Maksimum': rank, 'Rata-rata': rank, 'Rank': 0}])
            res = pd.concat([res, out])
    
    # Convert the 'Rank' column to numeric type (if it's not already)
    res['Rank'] = pd.to_numeric(res['Rank'])
    
    # Calculate the count of recommendations for each lecturer
    res['Count'] = res.groupby('Dosen')['Judul'].transform(lambda x: x.str.split(',').str.len())
    
    # Calculate the average rank for each lecturer
    res['Rata-rata'] = (res['Rata-rata'] / res['Count'])
    
    # Calculate the total rank (total of maximum and average) and combine them into 'Rank' column
    res['Rank'] = (res['Nilai Maksimum'] + res['Rata-rata'])/2
    
    # Sort the DataFrame based on the total rank in descending order
    res = res.sort_values(by='Rank', ascending=False)
    
    # Reset the index of the DataFrame
    res = res.reset_index(drop=True)
    
    return res

rekomendasi = get_recomendation("Media pembelajaran smk berbasis website")
rekomendasi


@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        query = request.json['query']
        rekomendasi = get_recomendation(query)
        rekomendasi_json = json.dumps(json.loads(rekomendasi.to_json(orient='records')), indent=4)
        return rekomendasi_json
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

    