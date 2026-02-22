import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split

RANDOM_STATE = 123456

def save_tweets_to_csv(input_file='data/tweets.txt', output_file='data/raw_dataset.csv'):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            tweets = infile.readlines()

        # Открываем CSV файл для записи
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)

            # Записываем заголовок
            writer.writerow(['tweet'])

            # Записываем каждый твит в отдельную строку
            for tweet in tweets:
                writer.writerow([tweet.strip()])  # Удаляем пробелы и переносы строк

        print(f"Данные успешно переформатированы в {output_file}")

    except Exception as e:
        print(f"Произошла ошибка при переформатировании данных: {e}")

def clean_text(text):
    # Очищаем датасет
    text = str(text).lower()
    text = re.sub(r'http[s]?://[^\s]+', '', text)   # ссылки
    text = re.sub(r'@[^\s]+', '', text)             # упоминания
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # спецсимволы
    text = re.sub(r'\s+', ' ', text).strip()        # лишние пробелы
    return text

def process_dataset():
    # Загружает, чистит и сохраняет датасет
    df = pd.read_csv("data/raw_dataset.csv")
    df['tweet'] = df['tweet'].astype(str).apply(clean_text)
    df = df[df['tweet'].str.len() > 1]
    df[['tweet']].to_csv("data/dataset_processed.csv", index=False)
    print("Очищенный датасет сохранен: data/dataset_processed.csv")

def split_dataset():
    # Разделение на train, val и test
    df = pd.read_csv("data/dataset_processed.csv")
    train, temp = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_STATE)

    train[['tweet']].to_csv("data/train.csv", index=False)
    val[['tweet']].to_csv("data/val.csv", index=False)
    test[['tweet']].to_csv("data/test.csv", index=False)

    print(f"\nДатасет размерностью {len(df)} строк разделен на:")
    print(f"train: {len(train)} \nval: {len(val)} \ntest: {len(test)}")