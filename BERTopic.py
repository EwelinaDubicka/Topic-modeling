import os
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.error("Plik z polskimi stopwords nie został znaleziony. Upewnij się, że plik istnieje.")
        return []

stopwords_polish = load_polish_stopwords()

# Inicjalizacja CountVectorizer z polskimi słowami stop do użycia w BERTopic
vectorizer_model = CountVectorizer(stop_words=stopwords_polish)

# Funkcja do podziału długiego tekstu na mniejsze fragmenty
def split_long_text(text, max_length=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 dla spacji
        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Funkcja do przetwarzania pojedynczego pliku CSV
def process_single_file(input_file_path):
    logging.info(f"Rozpoczynanie przetwarzania pliku {input_file_path}")
    
    # Wczytanie pliku CSV
    try:
        df = pd.read_csv(input_file_path, sep=',')
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania pliku {input_file_path}: {e}")
        return

    df['processed_text'] = df['processed_text'].fillna('')

    # Podziel tekst na fragmenty
    documents = []
    for text in df['processed_text']:
        documents.extend(split_long_text(text, max_length=500))

    # Inicjalizacja modelu BERTopic
    topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer_model)

    # Dopasowanie modelu do przetworzonych fragmentów tekstu
    logging.info(f"Trenowanie modelu BERTopic dla pliku {input_file_path}")
    
    try:
        topics, probabilities = topic_model.fit_transform(documents)
    except Exception as e:
        logging.error(f"Błąd podczas trenowania modelu dla pliku {input_file_path}: {e}")
        return

    # Zapisz wyniki do pliku CSV
    output_file_path = os.path.join(output_folder_path, f"processed_{os.path.basename(input_file_path)}")
    
    try:
        result_df = pd.DataFrame({"text_segment": documents, "topic": topics})
        result_df.to_csv(output_file_path, index=False)
        logging.info(f"Wyniki zapisane w {output_file_path}")
    except Exception as e:
        logging.error(f"Błąd podczas zapisywania wyników dla pliku {input_file_path}: {e}")

    # Tworzenie i zapisywanie wizualizacji
    try:
        # Mapa cieplna tematów
        heatmap_fig = topic_model.visualize_heatmap()
        heatmap_fig.write_html(os.path.join(output_folder_path, f"heatmap_{os.path.basename(input_file_path)}.html"))

        # Hierarchia tematów
        hierarchy_fig = topic_model.visualize_hierarchy()
        hierarchy_fig.write_html(os.path.join(output_folder_path, f"hierarchy_{os.path.basename(input_file_path)}.html"))

        # Rozkład tematów
        frequency_fig = topic_model.visualize_barchart()
        frequency_fig.write_html(os.path.join(output_folder_path, f"frequency_{os.path.basename(input_file_path)}.html"))

        logging.info(f"Wizualizacje dla {input_file_path} zostały zapisane")
    except Exception as e:
        logging.error(f"Błąd podczas tworzenia wizualizacji dla pliku {input_file_path}: {e}")

# Funkcja do przetwarzania wszystkich plików w folderze jednocześnie
def process_files_in_folder(input_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    input_files = [os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith('.csv')]

    if not input_files:
        logging.warning(f"Brak plików .csv w folderze {input_folder_path}")
        return

    logging.info(f"Rozpoczynanie przetwarzania {len(input_files)} plików CSV.")
    
    # Przetwarzanie plików równocześnie
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(process_single_file, input_files)

if __name__ == "__main__":
    process_files_in_folder(input_folder_path, output_folder_path)
    logging.info("Przetwarzanie wszystkich plików CSV zakończone.")
