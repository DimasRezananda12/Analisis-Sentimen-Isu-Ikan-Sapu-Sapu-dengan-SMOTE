import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# 1. Load Dataset
df = pd.read_csv('tweets-data/data_ikan_sapusapu.csv')
df = df.dropna(subset=['full_text'])

# 2. Text Preprocessing
print("\n[INFO] Memulai Preprocessing Teks...")
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

slang_dict = {
    "yg": "yang", "gk": "tidak", "g": "tidak", "bgt": "banget", 
    "bhya": "bahaya", "utk": "untuk", "jd": "jadi", "jgn": "jangan",
    "ga": "tidak", "gak": "tidak", "aja": "saja", "kalo": "kalau", 
    "klo": "kalau", "kl": "kalau", "tp": "tapi", "jg": "juga", 
    "gw": "saya", "w": "saya", "lu": "kamu", "udh": "sudah", 
    "udah": "sudah", "krn": "karena", "karna": "karena", "dr": "dari", 
    "kyk": "seperti", "kayak": "seperti", "kaya": "seperti", 
    "gimana": "bagaimana", "gmn": "bagaimana", "emg": "memang"
}

def bersihkan_teks(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#|[^a-zA-Z\s]', '', text)
    text = ' '.join([slang_dict.get(kata, kata) for kata in text.split()])
    text = stopword.remove(text)
    return stemmer.stem(text)

df['teks_bersih'] = df['full_text'].apply(bersihkan_teks)
print("[INFO] Preprocessing Selesai.\n")

# 3. Sentiment Labeling
print("[INFO] Memulai Pelabelan Sentimen...")
kata_negatif = [
    'bahaya', 'racun', 'limbah', 'mati', 'kotor', 'merkuri', 'sungai', 'penyakit', 
    'geli', 'seram', 'jangan', 'takut', 'ancam', 'punah', 'invasif', 'hama', 'buang', 
    'cemar', 'tercemar', 'parah', 'bau', 'amis', 'busuk', 'jijik', 'jelek', 'musnah', 
    'basmi', 'koruptor', 'korupsi', 'nakal', 'berat', 'logam', 'haram', 'mudarat', 'bunuh',
    'siksa', 'jahat', 'kasihan', 'ancaman', 'rugi'
]

kata_positif = [
    'enak', 'aman', 'bersih', 'protein', 'gurih', 'bakar', 'manfaat', 'murah', 'bagus', 
    'cantik', 'suka', 'cinta', 'estetik', 'berguna', 'pelihara', 'hias', 'pakan', 'pupuk',
    'cuan', 'halal', 'layak', 'konsumsi', 'untung', 'sehat', 'segar', 'laku', 'mahal'
]

def penentu_sentimen(text):
    skor = sum([1 for kata in text.split() if kata in kata_positif]) - sum([1 for kata in text.split() if kata in kata_negatif])
    return 'Positif' if skor > 0 else 'Negatif'

df['sentimen'] = df['teks_bersih'].apply(penentu_sentimen)
print("Distribusi Sentimen:")
print(df['sentimen'].value_counts())

# 4. Visualization & N-Gram Analysis
print("\n[INFO] Menyiapkan Visualisasi dan Analisis N-Gram...")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='sentimen', data=df, hue='sentimen', palette={'Negatif':'#d9534f', 'Positif':'#5cb85c'}, legend=False)
plt.title('Proporsi Sentimen Publik', fontsize=14)
plt.ylabel('Jumlah Cuitan')

plt.subplot(1, 2, 2)
teks_negatif = ' '.join(df[df['sentimen'] == 'Negatif']['teks_bersih'])
if teks_negatif.strip():
    wordcloud = WordCloud(width=500, height=300, background_color='white', colormap='Reds').generate(teks_negatif)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Topik Utama (Sentimen Negatif)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()

def get_top_ngrams(corpus, n=10, ngram_range=(1,1)):
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(words_freq, columns=['Teks', 'Frekuensi'])

pos_corpus = df[df['sentimen'] == 'Positif']['teks_bersih']
neg_corpus = df[df['sentimen'] == 'Negatif']['teks_bersih']

plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.barplot(x='Frekuensi', y='Teks', data=get_top_ngrams(pos_corpus, 10, (1,1)), hue='Teks', palette='Greens_r', legend=False)
plt.title('Top 10 Kata Tunggal (Positif)')

plt.subplot(2, 2, 2)
sns.barplot(x='Frekuensi', y='Teks', data=get_top_ngrams(neg_corpus, 10, (1,1)), hue='Teks', palette='Reds_r', legend=False)
plt.title('Top 10 Kata Tunggal (Negatif)')

plt.subplot(2, 2, 3)
sns.barplot(x='Frekuensi', y='Teks', data=get_top_ngrams(pos_corpus, 10, (2,2)), hue='Teks', palette='Greens_r', legend=False)
plt.title('Top 10 Frasa 2 Kata (Positif)')

plt.subplot(2, 2, 4)
sns.barplot(x='Frekuensi', y='Teks', data=get_top_ngrams(neg_corpus, 10, (2,2)), hue='Teks', palette='Reds_r', legend=False)
plt.title('Top 10 Frasa 2 Kata (Negatif)')

plt.tight_layout()
plt.show()

# 5. Model Training & Comparison dengan SMOTE & Matrix Evaluation
print("\n[INFO] Melatih dan Membandingkan Model Machine Learning...")
if len(df) > 20:
    X = TfidfVectorizer().fit_transform(df['teks_bersih'])
    y = df['sentimen']
    
    # Membagi data Ujian dan data Latihan (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Menggunakan SMOTE untuk menyeimbangkan data training (Negatif vs Positif)
    print(f"[INFO] Jumlah data sebelum SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Jumlah data sesudah SMOTE: {dict(y_train_smote.value_counts())}")

    models = {
        'Naive Bayes': MultinomialNB(),
        'Support Vector Machine (SVM)': SVC(kernel='linear'),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Neural Network (MLP)': MLPClassifier(max_iter=1000, random_state=42)
    }

    results = []
    trained_models = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_smote, y_train_smote) # Fit ke data yang sudah di-SMOTE
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results.append((name, acc))
            trained_models[name] = y_pred # Simpan prediksi untuk evaluasi lebih lanjut
        except Exception:
            results.append((name, 0.0))

    df_results = pd.DataFrame(results, columns=['Model', 'Akurasi'])
    df_results = df_results.sort_values(by='Akurasi', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Akurasi', y='Model', data=df_results, hue='Model', palette='viridis', legend=False)
    plt.title('Perbandingan Akurasi 7 Model Machine Learning (Setelah SMOTE)', fontsize=14)
    plt.xlim(0, 100)
    
    for index, value in enumerate(df_results['Akurasi']):
        plt.text(value + 1, index, f'{value:.2f}%', va='center')
        
    plt.tight_layout()
    plt.show()

    print("\n🏆 TOP 3 MODEL TERTINGGI:")
    print("-" * 45)
    for i in range(3):
        print(f"[{i+1}] {df_results.iloc[i]['Model']} : {df_results.iloc[i]['Akurasi']:.2f}%")
    print("-" * 45)

    # Menampilkan Evaluasi Matrix untuk Model Terbaik (Peringkat 1)
    best_model_name = df_results.iloc[0]['Model']
    best_y_pred = trained_models[best_model_name]

    print(f"\n[INFO] Evaluasi Mendalam untuk Model Terbaik ({best_model_name}):")
    print(classification_report(y_test, best_y_pred))

    # Confusion Matrix Visualization
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_y_pred, labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Tebakan AI (Predicted)')
    plt.ylabel('Kenyataan (Actual)')
    plt.tight_layout()
    plt.show()

else:
    print("[WARN] Data terlalu sedikit untuk melatih model.")