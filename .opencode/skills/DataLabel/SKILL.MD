---
name: AutoLabel Skill
description: Автоматическая разметка данных с генерацией спецификации, оценкой качества и экспортом в LabelStudio для ручной доразметки
license: MIT
compatibility: opencode
---
## About
Этот навык предназначен для автоматической разметки данных (текст, аудио или изображения) с генерацией подробной спецификации разметки, оценкой качества через сравнение с человеческой разметкой и экспортом задач для ручной доразметки в LabelStudio. **Иди строго по шагам!!**

**Файл с классами - topics.md**

## Структура данных
- `data/raw/` — исходные данные для разметки
- `data/processed/` — размеченные данные
- `data/labeling/` — спецификации, отчёты качества и экспорт для LabelStudio
- `data/labeling/specs/` — спецификации разметки
- `data/labeling/quality/` — метрики качества и сравнительные отчёты
- `data/labeling/labelstudio/` — JSON-файлы для импорта в LabelStudio

## What I do

### Шаг 1: Определение модальности и задачи
Спроси пользователя:
1. Сколько классов/категорий ожидается?
2. Есть ли примеры разметки для few-shot обучения?

### Шаг 2: Генерация спецификации разметки
Создай файл `data/labeling/specs/labeling_spec.md` со следующей структурой:

```markdown
# Спецификация разметки: [Название задачи]

## 1. Описание задачи
- Тип задачи: [классификация/NER/etc]
- Модальность: [text/audio/image]
- Цель: [что определяем]

## 2. Классы с определениями
| Класс | Определение | Критерии включения | Критерии исключения |
|-------|-------------|-------------------|---------------------|
| CLASS_1 | ... | ... | ... |
| CLASS_2 | ... | ... | ... |

## 3. Примеры разметки (минимум 3 на класс)
### CLASS_1
- Пример 1: "..." → CLASS_1 (пояснение)
- Пример 2: "..." → CLASS_1 (пояснение)
- Пример 3: "..." → CLASS_1 (пояснение)

### CLASS_2
...

## 4. Граничные случаи
| Ситуация | Решение | Обоснование |
|----------|---------|-------------|
| Неоднозначный контент | ... | ... |
| Множественные метки | ... | ... |
| Недостаточно данных | ... | ... |

## 5. Инструкция для разметчика
- [Пошаговая инструкция]
- [Что делать при сомнениях]
- [Контакты для вопросов]
```

**Обязательно покажи спецификацию пользователю и получи одобрение перед продолжением!**

### Шаг 3: Автоматическая разметка
В зависимости от модальности:

**Для текста:**
```bash
# Установка зависимостей
.venv/bin/python -m pip install transformers datasets scikit-learn

# Запуск авторазметки
.venv/bin/python .opencode/skills/AutoLabel/scripts/auto_label.py     --input data/processed/processed.csv     --output data/processed/labeled_data.csv     --modality text     --task classification     --spec data/labeling/specs/labeling_spec.md
```

**Для изображений:**
```bash
.venv/bin/python .opencode/skills/AutoLabel/scripts/auto_label.py     --input data/processed/images/     --output data/processed/labeled_images.csv     --modality image     --task classification     --spec data/labeling/specs/labeling_spec.md
```

**Требования к скрипту авторазметки:**
- Использовать zero-shot или few-shot подход (Hugging Face transformers (легковесная модель))
- Сохранять confidence score для каждой метки (0-1)
- Добавить колонки: `auto_label`, `confidence`, `model_version`
- Обработка ошибок и логирование

### Шаг 4: Human-in-the-loop (флаги низкой уверенности)
Автоматически создать файл для ручной разметки:

```python
import pandas as pd

# Загрузить размеченные данные
df = pd.read_csv('data/processed/labeled_data.csv')

# Порог уверенности (по умолчанию 0.7, можно переопределить)
CONFIDENCE_THRESHOLD = 0.7

# Отфильтровать примеры с низкой уверенностью
low_confidence = df[df['confidence'] < CONFIDENCE_THRESHOLD].copy()

# Сохранить для ручной разметки
low_confidence.to_csv('data/labeling/labelstudio/low_confidence_tasks.csv', index=False)

print(f"Найдено {len(low_confidence)} примеров с confidence < {CONFIDENCE_THRESHOLD}")
print(f"Процент от общего датасета: {len(low_confidence)/len(df)*100:.1f}%")
```

**Покажи пользователю:**
- Сколько примеров отфлажено
- Распределение по классам среди низкой уверенности
- Примеры таких случаев

### Шаг 5: Экспорт в LabelStudio
Создать JSON-файл для импорта в LabelStudio:

```python
import json
import pandas as pd

def export_to_labelstudio(input_csv, output_json, project_config):
    """
    Генерация JSON для LabelStudio
    """
    df = pd.read_csv(input_csv)

    tasks = []
    for idx, row in df.iterrows():
        task = {
            "id": idx,
            "data": {
                "text": row.get('text', ''),
                "image": row.get('image_url', ''),
                "audio": row.get('audio_url', '')
            },
            "annotations": [],
            "predictions": [{
                "model_version": row.get('model_version', 'auto-v1'),
                "score": float(row.get('confidence', 0)),
                "result": [{
                    "from_name": "label",
                    "to_name": project_config['target'],
                    "type": "choices",
                    "value": {
                        "choices": [row['auto_label']]
                    }
                }]
            }]
        }
        tasks.append(task)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    return len(tasks)

# Конфигурация проекта
config = {
    "target": "text"
}

# Экспорт основного датасета
export_to_labelstudio(
    'data/processed/labeled_data.csv',
    'data/labeling/labelstudio/all_tasks.json',
    config
)

# Экспорт низкой уверенности (отдельный проект)
if len(low_confidence) > 0:
    export_to_labelstudio(
        'data/labeling/labelstudio/low_confidence_tasks.csv',
        'data/labeling/labelstudio/low_confidence_tasks.json',
        config
    )
```

**Проверка JSON:**
- Валидация структуры (id, data, annotations/predictions)
- Проверка на дубли ID
- Тестовая загрузка в LabelStudio (опционально)

### Шаг 6: Оценка качества (требует ручной разметки)
Сам разметь выборку и сравни её с авторазметкой:

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_quality(auto_labels_path, human_labels_path, output_dir):
    """
    Сравнение авторазметки с человеческой
    """
    # Загрузка данных
    auto_df = pd.read_csv(auto_labels_path)
    human_df = pd.read_csv(human_labels_path)

    # Объединение по ID
    merged = auto_df.merge(human_df, on='id', suffixes=('_auto', '_human'))

    # Метрики
    kappa = cohen_kappa_score(merged['label_auto'], merged['label_human'])

    # Процент согласия
    agreement = (merged['label_auto'] == merged['label_human']).mean() * 100

    # Classification report
    report = classification_report(
        merged['label_human'], 
        merged['label_auto'],
        output_dict=True
    )

    # Распределение меток
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Auto labels distribution
    auto_counts = merged['label_auto'].value_counts()
    axes[0].bar(auto_counts.index, auto_counts.values, color='skyblue', alpha=0.7)
    axes[0].set_title('Распределение: Авторазметка')
    axes[0].set_ylabel('Количество')

    # Human labels distribution
    human_counts = merged['label_human'].value_counts()
    axes[1].bar(human_counts.index, human_counts.values, color='lightcoral', alpha=0.7)
    axes[1].set_title('Распределение: Человеческая разметка')
    axes[1].set_ylabel('Количество')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/label_distribution_comparison.png')

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(merged['label_human'], merged['label_auto'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Авторазметка')
    plt.ylabel('Человеческая разметка')
    plt.title('Матрица ошибок')
    plt.savefig(f'{output_dir}/confusion_matrix.png')

    # Отчёт
    with open(f'{output_dir}/quality_report.txt', 'w') as f:
        f.write(f"""# Отчёт о качестве авторазметки

## Метрики согласия
- Cohen's Kappa: {kappa:.3f}
- Процент согласия: {agreement:.1f}%

## Распределение меток
### Авторазметка:
{auto_counts.to_string()}

### Человеческая разметка:
{human_counts.to_string()}

## Classification Report
""")
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"
{cls}:
")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.3f}
")

    return {
        'kappa': kappa,
        'agreement': agreement,
        'report': report
    }

# Запуск оценки
results = evaluate_quality(
    'data/processed/labeled_data.csv',
    'data/labeling/human_labels.csv',
    'data/labeling/quality/'
)
```

**Покажи пользователю:**
- Cohen's Kappa (цель: > 0.6 — хорошо, > 0.8 — отлично)
- Процент согласия
- Классы с наибольшими расхождениями
- Рекомендации по улучшению

## Требования к выходным файлам

| Файл | Описание | Обязательный |
|------|----------|--------------|
| `data/labeling/specs/labeling_spec.md` | Спецификация разметки | Да |
| `data/processed/labeled_data.csv` | Данные с авторазметкой | Да |
| `data/labeling/labelstudio/all_tasks.json` | Все задачи для LabelStudio | Да |
| `data/labeling/labelstudio/low_confidence_tasks.json` | Низкая уверенность (HITL) | Бонус |
| `data/labeling/quality/quality_report.txt` | Отчёт о качестве | После ручной разметки |
| `data/labeling/quality/confusion_matrix.png` | Матрица ошибок | После ручной разметки |

## После каждого шага спрашивай пользователя, хочет ли он продолжить к следую<response clipped>