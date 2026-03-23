---
name: Data Collection Skill
description: Собирает датасеты по новостям для классификации темы и саммари из Hugging Face и Интернета
license: MIT
compatibility: opencode
---
## About
Этот навык предназначен для сбора данных по новостям для классификации темы и саммари из различных источников, таких как Hugging Face и веб-скрапинг. Он может загружать датасеты, извлекать тексты и аннотации, а также предоставлять предварительный просмотр данных. Не забывай удалять временные файлы. **Иди строго по шагам!! ВСЕГДА СПРАШИВАЙ ПОЛЬЗОВАТЕЛЯ ПЕРЕЙТИ ЛИ К СЛЕДУЮЩЕМУ ШАГУ**
## What I do
1. Создай папки

```
# Создание структуры папок
mkdir -p data/processed
mkdir -p data/eda
mkdir -p data/raw
```

# 1. Поиск
Спроси пользователя, где искать новости и для какой задачи
Гайд по использованию скрипта для поиска и загрузки датасетов по сегментации автомобилей:

| Action | Command |
|--------|---------|
| Search Hugging Face | `.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py huggingface search "sentiment"` |
| Download HF dataset | `.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py huggingface download "dataset-name"` |
| Preview dataset | `.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py preview dataset.csv` |
| Generate data card | `.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py datacard dataset.csv --output README.md` |
| List local datasets | `.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py list` |

**Пример поиска датасетов по сегментации**
.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py huggingface search "news english"

**Обязательно после поиска hf датасета предложи веб скрпинг**
### 2. Web Scraping

```bash
# Search the web with DuckDuckGo
bash -c '''source .venv/bin/activate && crawl4ai-skill search "python web scraping"'''
```

Для скраппинга используй библиотеку crawl4ai-skill
```bash
# Scrape a single web page
bash -c "source .venv/bin/activate && crawl4ai-skill crawl https://example.com"
```
```bash
# Crawl entire website / spider
bash -c "source .venv/bin/activate && crawl4ai-skill crawl-site https://docs.python.org --max-pages 50"
```

Или пиши скрипты по скрапингу на bs4. Важно чтобы кол-во совпадало с тем что написал пользователь

# 3. Таблица для пользователя
Показать таблицу с ОБЯЗАТЕЛЬНЫМ разделением:
```
## Найденные источники для "news in iran"

### Готовые датасеты:
| # | Название | Источник | Размер | Примечание |
|---|----------|----------|--------|------------|
| 1 | news | HuggingFace | 500 изобр. | есть маски |

### Сайты для скрапинга:
| # | URL | Что парсить | ~Размер | Примечание |
|---|-----|-------------|---------|------------|
| 3 | news.com/car | news | >1000 |  |
| 4 | news.com/photos | news | >500 ||
```

# 4. Подтверждение
ОБЯЗАТЕЛЬНО спросить:

Найдено X источников. Какие скачиваем/парсим? Укажи номера через запятую.

Ждать ответа пользователя.

# 5. Скачивание
Для Hugging Face:
```
.venv/bin/python .opencode/skills/DataCollection/scripts/dataset.py huggingface download "dataset-name" --output-dir data/raw
```

Для скраппинга используй библиотеку crawl4ai-skill
```bash
# Scrape a single web page
bash -c "source .venv/bin/activate && crawl4ai-skill crawl https://example.com"
```

Или пиши скрипты с использование bs4, time.sleep() и т.д. для парсинга контента новостных ссылок.
Не вставляй тему и саммари, если их нет в данных, просто сохраняй текст и источник

# 6. Унифицируй скачанные датасеты:
Датасеты из HF и скрапингаs должны иметь 3 столбца (либо 2, если нет лейблов): **text**, **theme**, **summary** **source**, **meta**. В **meta**
добавляй данные, из других столбцов кроме первых двух, если они существуют. Собери один датасет и сохрани его в `data/processed/processed.csv`. Не добавляй **summary** и **theme** если они неизвестны

# 7. EDA
Необходимо провести EDA в папке `data/eda`. EDA: распределение классов, длины текстов, топ-20 слов, пропуски и тд. Сохраняй графики в папке. Шаги выполнения:

1. **Загрузка и валидация данных:**
   - Загрузить данные из CSV-файла (используя pandas).
   - Проверить наличие пропусков (null/NaN) в ключевых полях: `text`, `theme`, `summary`, `source`, `meta`.
   - Проверить типы данных и базовую статистику.
2. **Анализ категорий (Классы):**
   - Рассчитать распределение по полю `theme`.
   - Построить bar-plot (столбчатую диаграмму), отображающую количество записей для каждой темы.
   - Сохранить график как `theme_distribution.png` в папку `[ПУТЬ_К_ПАПКЕ_EDA]`.
3. **Анализ длины текстов:**
   - Рассчитать количество слов и количество символов для поля `text`.
   - Построить гистограмму распределения длин текстов (в словах).
   - Сохранить график как `word_count_distribution.png` в папку `[ПУТЬ_К_ПАПКЕ_EDA]`.
4. **Текстовый анализ (Топ-20 слов):**
   - Очистить текст (привести к нижнему регистру, удалить пунктуацию, удалить стоп-слова для русского/амхарского языков).
   - Вычислить 20 самых часто встречающихся слов во всем корпусе.
   - Построить bar-plot для топ-20 слов.
   - Сохранить график как `top_20_words.png` в папку `[ПУТЬ_К_ПАПКЕ_EDA]`.
5. **Генерация отчета:**
   - Сформировать текстовый файл `eda_summary.txt` в папке `[ПУТЬ_К_ПАПКЕ_EDA]`.
   - В отчете указать:
     - Общее количество записей.
     - Количество пропусков по колонкам.
     - Описание распределения тем.
     - Среднюю/медиану длины текстов.
     - Список топ-20 слов с их частотами.
6. **Генерация ноутбука:**
    - Сгенерируй data/notebooks/eda.ipynb — где ьудет код EDA и визуализации
#### Ограничения:
- Если данные не содержат аудио — пропустить анализ длительности аудио и спектрограмм.
- Если возникают проблемы с отрисовкой шрифтов (например, для амхарских символов), использовать fontconfig или подставить шрифт, поддерживающий юникод, чтобы графики были читаемы.
- Работать строго в рамках предоставленного пути к файлам.

