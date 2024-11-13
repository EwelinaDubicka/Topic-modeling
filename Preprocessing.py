import os
import pandas as pd
import spacy
import re
import string
import concurrent.futures

# Załaduj model spaCy dla języka polskiego
nlp = spacy.load("pl_core_news_lg")

# Ścieżki folderów
input_folder_path = r'your_path'
output_folder_path = r'your_path'

# Wczytaj polskie stopwords z pliku
def load_polish_stopwords():
    try:
        with open(r'your_path', 'r', encoding='utf-8') as f:
            stopwords_polish = f.read().splitlines()
        stopwords_polish = [word.lower().strip() for word in stopwords_polish]
        return stopwords_polish
    except FileNotFoundError:
        print("Plik z polskimi stopwords nie został znaleziony.")
        return []

# Wczytaj polskie stopwords
stopwords_polish = load_polish_stopwords()

# Funkcja do przetwarzania tekstu
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+(?:-\d+)?', '', text)
    text = re.sub(r'\b[mcdlxvi]+\b', '', text)
    text = re.sub(r'\d+\s*½|\d+/\d+|½', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    doc = nlp(text)
    
    filtered_text = [
        token.lemma_ for token in doc if token.lemma_ not in stopwords_polish and not token.is_punct
    ]
    return ' '.join(filtered_text)

# Funkcja do przetwarzania pojedynczego pliku CSV
def process_single_file(input_file_path, output_file_path):
    # Wczytaj plik CSV bez nagłówków
    df = pd.read_csv(input_file_path, sep=',', header=None)
    
    # Zmień nazwę pierwszej kolumny na 'text'
    df.rename(columns={0: 'text'}, inplace=True)
    
    # Przetwórz tekst, jeśli kolumna 'text' istnieje
    if 'text' in df.columns:
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Zapisz przetworzony DataFrame do nowego pliku CSV
        df.to_csv(output_file_path, index=False)
        print(f"Processed file saved: {output_file_path}")
    else:
        print(f"Brak kolumny 'text' w pliku {input_file_path}")

# Funkcja do przetwarzania wszystkich plików CSV w folderze
def process_files_in_folder(input_folder_path, output_folder_path):
    # Upewnij się, że folder wyjściowy istnieje
    os.makedirs(output_folder_path, exist_ok=True)

    input_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    if not input_files:
        print(f"No .csv files found in {input_folder_path}")
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in input_files:
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, f"processed_{filename}")
            futures.append(executor.submit(process_single_file, input_file_path, output_file_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    process_files_in_folder(input_folder_path, output_folder_path)
    print("Przetwarzanie wszystkich plików CSV zakończone.")
