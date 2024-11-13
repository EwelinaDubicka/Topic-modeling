import os
import pandas as pd
import re
import concurrent.futures

# Ścieżka do folderu z plikami txt
input_folder_path = r"your_path"
output_folder_path = r"your_path"

# Funkcja do dzielenia tekstu na zdania
def split_into_sentences(text):
    # Wyrażenie regularne ignorujące kropki po pojedynczych literach (np. "A.", "B.")
    sentences = re.split(r'(?<!\b\w\b)(?<!\b\w\.\b)(?<=\.|\?|!)\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Funkcja do przetwarzania pojedynczego pliku txt
def process_single_file(input_file_path, output_file_path):
    data = []
    
    # Wczytaj plik txt i podziel na zdania
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences = split_into_sentences(line)
            data.extend([[sentence] for sentence in sentences])  # Każde zdanie jako osobny wiersz

    # Konwersja do DataFrame
    df = pd.DataFrame(data)

    # Zapisz do pliku CSV
    df.to_csv(output_file_path, index=False, header=False)
    print(f"Processed file: {input_file_path} -> {output_file_path}")

# Funkcja do przetwarzania wszystkich plików txt w folderze
def process_files_in_folder(input_folder_path, output_folder_path):
    # Sprawdzenie istnienia folderów
    assert os.path.exists(input_folder_path), f"Input folder not found: {input_folder_path}"
    os.makedirs(output_folder_path, exist_ok=True)

    # Zbieranie plików .txt do przetworzenia
    input_files = [f for f in os.listdir(input_folder_path) if f.endswith('.txt')]
    if not input_files:
        print(f"No .txt files found in {input_folder_path}")
        return

    # Równoległe przetwarzanie plików
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for filename in input_files:
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}.csv")
            futures.append(executor.submit(process_single_file, input_file_path, output_file_path))

        # Czekaj na zakończenie wszystkich zadań
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Możesz sprawdzić ewentualne wyjątki

if __name__ == "__main__":
    process_files_in_folder(input_folder_path, output_folder_path)
    print("Przetwarzanie wszystkich plików txt do csv zakończone.")
