import os
import subprocess
import pandas as pd

print("=== TAHAP 1: SCRAPING DATA TWITTER ===")

twitter_auth_token = "9bd25174e775c9e470897d0786b5ead805f2628c" 
search_keyword = "ikan sapu sapu OR ikan pembersih kaca"
limit = 200
filename = "data_ikan_sapusapu.csv"

os.environ['TWITTER_AUTH_TOKEN'] = twitter_auth_token

command = f'npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" -l {limit} --token {twitter_auth_token}'

print(f"Mengambil data untuk keyword: '{search_keyword}'...")

try:
    subprocess.run(command, shell=True, check=True)
    print("Scraping Selesai!")
except subprocess.CalledProcessError as e:
    print(f"Terjadi kesalahan saat scraping: {e}")

path_data = f"tweets-data/{filename}"
if os.path.exists(path_data):
    df = pd.read_csv(path_data)
    df = df[['full_text']].dropna().drop_duplicates()
    print(f"Berhasil memuat {len(df)} tweet unik.")
else:
    print("File CSV tidak ditemukan, pastikan proses scraping berhasil.")