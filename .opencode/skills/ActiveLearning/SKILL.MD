name: ActiveLearning Skill
description: Реализует Active Learning цикл для умного отбора данных: старт с N=50, 5 итераций, сравнение стратегий entropy vs random, оценка экономии данных
license: MIT
compatibility: opencode
---
## About
Этот навык реализует полный Active Learning (AL) цикл для умного отбора данных. Агент выполняет итеративное обучение, сравнивает стратегии отбора (entropy vs random), строит кривые обучения и оценивает, сколько примеров сэкономлено при том же качестве. Финальный агент интегрируется в пайплайн как `active_learning_op`. **Иди строго по шагам!!**

## Расположение данных:
- `data/raw/` — исходный пул неразмеченных данных
- `data/al_labeled/` — размеченные данные по итерациям (iteration_0, iteration_1, ...)
- `data/al_results/` — результаты экспериментов, метрики, кривые обучения

## What I do

### Шаг 0: Создание структуры папок
```bash
mkdir -p data/al_labeled data/al_results agents notebooks data/al_results/plots
```

### Шаг 1: Подготовка данных
Спроси пользователя:
1. **Путь к данным**: CSV файл с колонками (text/image_path/audio_path + label)
2. **Модальность**: text
3. **Тип задачи**: classification
4. **Размер начальной выборки**: по умолчанию N=50
5. **Бюджет на итерацию**: сколько примеров размечаем за раз (по умолчанию 20)
6. **Количество итераций**: по умолчанию 5

Загрузи данные и раздели на:
- `X_pool` — неразмеченный пул (все данные минус начальная выборка)
- `X_labeled`, `y_labeled` — начальная размеченная выборка (N=50 случайных примеров)
- `X_test`, `y_test` — тестовый сет (20% данных, фиксированный на все итерации)

### Шаг 2: Реализация стратегий отбора

**Стратегия 1: Entropy Sampling (Uncertainty-based)**
```python
import numpy as np
from scipy.stats import entropy

def entropy_sampling(probas):
    """
    Calculate entropy for each prediction.
    H(p) = -sum(p_i * log(p_i))
    Higher entropy = more uncertainty = more valuable for labeling
    """
    return np.array([entropy(p) for p in probas])

def select_entropy_samples(learner, X_pool, n_samples=20):
    """
    Select samples with highest prediction entropy.
    """
    probas = learner.predict_proba(X_pool)
    entropies = entropy_sampling(probas)
    # Выбираем индексы с максимальной энтропией
    selected_idx = np.argsort(entropies)[-n_samples:][::-1]
    return selected_idx, entropies[selected_idx]
```

**Стратегия 2: Random Sampling (Baseline)**
```python
def select_random_samples(X_pool, n_samples=20, seed=None):
    """
    Random selection baseline.
    """
    rng = np.random.RandomState(seed)
    n_pool = len(X_pool)
    selected_idx = rng.choice(n_pool, size=n_samples, replace=False)
    return selected_idx
```

### Шаг 3: Active Learning цикл

Реализуй AL цикл для каждой стратегии:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json

def run_active_learning(
    X_full, y_full,
    strategy='entropy',  # 'entropy' или 'random'
    n_initial=50,
    n_per_iteration=20,
    n_iterations=5,
    random_state=42
):
    """
    Run Active Learning loop.
    
    Returns:
        history: список метрик по каждой итерации
        final_model: обученная модель
        total_labeled: общее количество размеченных примеров
    """
    # Разделение данных
    from sklearn.model_selection import train_test_split
    X_pool_full, X_test, y_pool_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=random_state, stratify=y_full
    )
    
    # Начальная разметка (random N=50)
    rng = np.random.RandomState(random_state)
    initial_idx = rng.choice(len(X_pool_full), size=n_initial, replace=False)
    
    X_labeled = X_pool_full[initial_idx].copy()
    y_labeled = y_pool_full[initial_idx].copy()
    
    # Удаляем из пула
    X_pool = np.delete(X_pool_full, initial_idx, axis=0)
    y_pool = np.delete(y_pool_full, initial_idx, axis=0)
    
    # История метрик
    history = []
    
    # Начальное обучение
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_labeled, y_labeled)
    
    # Оценка на тесте
    y_pred = model.predict(X_test)
    history.append({
        'iteration': 0,
        'n_labeled': len(X_labeled),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'strategy': strategy
    })
    
    # AL итерации
    for iteration in range(1, n_iterations + 1):
        if strategy == 'entropy':
            # Entropy sampling
            probas = model.predict_proba(X_pool)
            entropies = entropy_sampling(probas)
            selected_idx = np.argsort(entropies)[-n_per_iteration:][::-1]
        else:
            # Random sampling
            rng = np.random.RandomState(random_state + iteration)
            selected_idx = rng.choice(len(X_pool), size=n_per_iteration, replace=False)
        
        # Добавляем размеченные примеры
        X_labeled = np.vstack([X_labeled, X_pool[selected_idx]])
        y_labeled = np.concatenate([y_labeled, y_pool[selected_idx]])
        
        # Удаляем из пула
        X_pool = np.delete(X_pool, selected_idx, axis=0)
        y_pool = np.delete(y_pool, selected_idx, axis=0)
        
        # Переобучаем модель
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_labeled, y_labeled)
        
        # Оценка
        y_pred = model.predict(X_test)
        history.append({
            'iteration': iteration,
            'n_labeled': len(X_labeled),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'strategy': strategy
        })
        
        # Сохраняем разметку итерации
        np.savez(f'data/al_labeled/{strategy}_iteration_{iteration}.npz',
                 X=X_labeled, y=y_labeled, 
                 new_idx=selected_idx, iteration=iteration)
    
    return history, model, len(X_labeled)
```

### Шаг 4: Запуск эксперимента и сравнение

Запусти обе стратегии и сравни:

```python
# Запуск Entropy
history_entropy, model_entropy, n_labeled_entropy = run_active_learning(
    X_full, y_full, strategy='entropy', n_initial=50, n_per_iteration=20, n_iterations=5
)

# Запуск Random
history_random, model_random, n_labeled_random = run_active_learning(
    X_full, y_full, strategy='random', n_initial=50, n_per_iteration=20, n_iterations=5
)

# Сохранение результатов
results = {
    'entropy': {
        'history': history_entropy,
        'final_accuracy': history_entropy[-1]['accuracy'],
        'final_f1': history_entropy[-1]['f1_macro']
    },
    'random': {
        'history': history_random,
        'final_accuracy': history_random[-1]['accuracy'],
        'final_f1': history_random[-1]['f1_macro']
    }
}

with open('data/al_results/experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Шаг 5: Визуализация кривых обучения

Создай график сравнения на одном plot:

```python
import matplotlib.pyplot as plt

def plot_learning_curves(history_entropy, history_random, save_path='data/al_results/plots/learning_curves.png'):
    """
    Plot learning curves: accuracy vs number of labeled samples.
    Both strategies on same graph.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Данные
    n_labeled_entropy = [h['n_labeled'] for h in history_entropy]
    acc_entropy = [h['accuracy'] for h in history_entropy]
    f1_entropy = [h['f1_macro'] for h in history_entropy]
    
    n_labeled_random = [h['n_labeled'] for h in history_random]
    acc_random = [h['accuracy'] for h in history_random]
    f1_random = [h['f1_macro'] for h in history_random]
    
    # Accuracy
    axes[0].plot(n_labeled_entropy, acc_entropy, 'b-o', label='Entropy Sampling', linewidth=2, markersize=8)
    axes[0].plot(n_labeled_random, acc_random, 'r--s', label='Random Sampling', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Labeled Samples', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Learning Curves: Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1].plot(n_labeled_entropy, f1_entropy, 'b-o', label='Entropy Sampling', linewidth=2, markersize=8)
    axes[1].plot(n_labeled_random, f1_random, 'r--s', label='Random Sampling', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Labeled Samples', fontsize=12)
    axes[1].set_ylabel('F1 Macro', fontsize=12)
    axes[1].set_title('Learning Curves: F1 Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path
```

### Шаг 6: Расчёт экономии данных

```python
def calculate_savings(history_entropy, history_random, target_metric='accuracy'):
    """
    Calculate how many samples are saved by entropy vs random
    to achieve the same quality.
    
    Returns:
        savings: количество сэкономленных примеров
        savings_percent: процент экономии
        comparison_table: таблица сравнения
    """
    # Находим финальное качество entropy
    final_acc_entropy = history_entropy[-1][target_metric]
    final_n_entropy = history_entropy[-1]['n_labeled']
    
    # Находим, сколько нужно random для достижения того же качества
    n_needed_random = None
    for h in history_random:
        if h[target_metric] >= final_acc_entropy * 0.98:  # 98% от целевого качества
            n_needed_random = h['n_labeled']
            break
    
    if n_needed_random is None:
        n_needed_random = history_random[-1]['n_labeled']
    
    savings = n_needed_random - final_n_entropy
    savings_percent = (savings / n_needed_random) * 100
    
    comparison_table = {
        'metric': target_metric,
        'target_quality': final_acc_entropy,
        'entropy_samples': final_n_entropy,
        'random_samples_needed': n_needed_random,
        'samples_saved': savings,
        'savings_percent': savings_percent
    }
    
    return savings, savings_percent, comparison_table
```

### Шаг 7: Генерация отчёта

Создай `data/al_results/al_report.md`:

```markdown
# Active Learning Experiment Report

## Setup
- Initial labeled samples: 50
- Samples per iteration: 20
- Number of iterations: 5
- Final labeled (entropy): {N_entropy}
- Final labeled (random): {N_random}

## Results

### Final Performance
| Strategy | Accuracy | F1 Macro | Samples Used |
|----------|----------|----------|--------------|
| Entropy  | {acc_entropy:.3f} | {f1_entropy:.3f} | {n_entropy} |
| Random   | {acc_random:.3f} | {f1_random:.3f} | {n_random} |

### Data Savings
To achieve {target_acc:.3f} accuracy:
- Entropy sampling used: {n_entropy} samples
- Random sampling needs: ~{n_random_equiv} samples
- **Savings: {savings} samples ({savings_percent:.1f}%)**

### Learning Curves
![Learning Curves](plots/learning_curves.png)

### Conclusion
Entropy-based Active Learning позволяет достичь того же качества 
на {savings_percent:.1f}% меньше размеченных данных по сравнению 
со случайным отбором.
```

### Шаг 8: Создание финального агента

Создай файл `agents/al_agent.py`:

```python
"""
Active Learning Agent для интеграции в финальный пайплайн.
"""

import numpy as np
from sklearn.base import BaseEstimator
from scipy.stats import entropy
import json
from typing import Tuple, List, Dict, Optional

class ActiveLearningAgent:
    """
    Active Learning агент для умного отбора данных.
    
    Usage:
        agent = ActiveLearningAgent(
            estimator=RandomForestClassifier(),
            n_initial=50,
            n_per_iteration=20,
            n_iterations=5,
            strategy='entropy'  # или 'random'
        )
        
        history = agent.fit(X_pool, y_pool_available)
        # y_pool_available — размечается только выбранные сэмплы
    """
    
    def __init__(
        self,
        estimator: BaseEstimator,
        n_initial: int = 50,
        n_per_iteration: int = 20,
        n_iterations: int = 5,
        strategy: str = 'entropy',
        random_state: int = 42,
        metric: str = 'accuracy'
    ):
        self.estimator = estimator
        self.n_initial = n_initial
        self.n_per_iteration = n_per_iteration
        self.n_iterations = n_iterations
        self.strategy = strategy
        self.random_state = random_state
        self.metric = metric
        
        self.history_: List[Dict] = []
        self.model_ = None
        self.X_labeled_ = None
        self.y_labeled_ = None
        
    def _entropy_sampling(self, probas: np.ndarray, n_samples: int) -> np.ndarray:
        """Select samples with highest entropy."""
        entropies = np.array([entropy(p) for p in probas])
        return np.argsort(entropies)[-n_samples:][::-1]
    
    def _random_sampling(self, n_pool: int, n_samples: int, seed: int) -> np.ndarray:
        """Random sampling."""
        rng = np.random.RandomState(seed)
        return rng.choice(n_pool, size=n_samples, replace=False)
    
    def fit(self, X_pool: np.ndarray, oracle) -> List[Dict]:
        """
        Run Active Learning loop.
        
        Args:
            X_pool: неразмеченный пул данных
            oracle: callable(X) -> y — функция разметки (или уже размеченные y)
        
        Returns:
            history: список метрик по итерациям
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        rng = np.random.RandomState(self.random_state)
        
        # Начальная выборка
        initial_idx = rng.choice(len(X_pool), size=self.n_initial, replace=False)
        
        # Получаем разметку для начальной выборки
        if callable(oracle):
            y_labeled = np.array([oracle(X_pool[i]) for i in initial_idx])
        else:
            y_labeled = oracle[initial_idx]
            
        X_labeled = X_pool[initial_idx].copy()
        X_remaining = np.delete(X_pool, initial_idx, axis=0)
        
        if not callable(oracle):
            y_remaining = np.delete(oracle, initial_idx, axis=0)
        
        # Обучение начальной модели
        self.model_ = self.estimator.fit(X_labeled, y_labeled)
        
        # История
        self.history_ = [{
            'iteration': 0,
            'n_labeled': len(X_labeled),
            'metric': None  # Нужен тестовый сет для оценки
        }]
        
        # AL итерации
        for iteration in range(1, self.n_iterations + 1):
            # Выбор стратегии
            if self.strategy == 'entropy':
                probas = self.model_.predict_proba(X_remaining)
                selected_idx = self._entropy_sampling(probas, self.n_per_iteration)
            else:
                selected_idx = self._random_sampling(
                    len(X_remaining), self.n_per_iteration, 
                    self.random_state + iteration
                )
            
            # Разметка выбранных примеров
            X_new = X_remaining[selected_idx]
            if callable(oracle):
                y_new = np.array([oracle(X_remaining[i]) for i in selected_idx])
            else:
                y_new = y_remaining[selected_idx]
            
            # Обновление labeled set
            X_labeled = np.vstack([X_labeled, X_new])
            y_labeled = np.concatenate([y_labeled, y_new])
            
            # Удаление из пула
            X_remaining = np.delete(X_remaining, selected_idx, axis=0)
            if not callable(oracle):
                y_remaining = np.delete(y_remaining, selected_idx, axis=0)
            
            # Переобучение
            self.model_ = self.estimator.fit(X_labeled, y_labeled)
            
            self.history_.append({
                'iteration': iteration,
                'n_labeled': len(X_labeled),
                'strategy': self.strategy
            })
        
        self.X_labeled_ = X_labeled
        self.y_labeled_ = y_labeled
        
        return self.history_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using trained model."""
        if self.model_ is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model_.predict(X)
    
    def save(self, path: str):
        """Save agent state."""
        import joblib
        state = {
            'history': self.history_,
            'model': self.model_,
            'X_labeled': self.X_labeled_,
            'y_labeled': self.y_labeled_,
            'params': {
                'n_initial': self.n_initial,
                'n_per_iteration': self.n_per_iteration,
                'n_iterations': self.n_iterations,
                'strategy': self.strategy
            }
        }
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: str):
        """Load agent state."""
        import joblib
        state = joblib.load(path)
        agent = cls(estimator=state['model'])
        agent.history_ = state['history']
        agent.model_ = state['model']
        agent.X_labeled_ = state['X_labeled']
        agent.y_labeled_ = state['y_labeled']
        return agent


# Функция для сравнения стратегий
def compare_strategies(
    X_full, y_full,
    estimator,
    n_initial=50,
    n_per_iteration=20,
    n_iterations=5,
    random_state=42
) -> Dict:
    """
    Compare entropy vs random sampling strategies.
    
    Returns:
        dict с результатами сравнения и экономией данных
    """
    from sklearn.model_selection import train_test_split
    
    # Разделение
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=random_state, stratify=y_full
    )
    
    # Entropy
    agent_entropy = ActiveLearningAgent(
        estimator=estimator,
        n_initial=n_initial,
        n_per_iteration=n_per_iteration,
        n_iterations=n_iterations,
        strategy='entropy',
        random_state=random_state
    )
    
    # Random
    agent_random = ActiveLearningAgent(
        estimator=estimator,
        n_initial=n_initial,
        n_per_iteration=n_per_iteration,
        n_iterations=n_iterations,
        strategy='random',
        random_state=random_state
    )
    
    # Запуск (симуляция — используем y_pool как oracle)
    history_entropy = []
    history_random = []
    
    # Реализация с полной оценкой на тесте
    # ... (полная реализация из Шага 3)
    
    return {
        'entropy': {'history': history_entropy},
        'random': {'history': history_random},
        'savings': None,  # рассчитать
        'savings_percent': None
    }
```

### Шаг 9: Создание ноутбука эксперимента

Создай `notebooks/al_experiment.ipynb` со следующей структурой:

```markdown
# Active Learning Experiment

## 1. Setup
- Загрузка данных
- Параметры эксперимента: N=50, 5 итераций, 20 samples/iteration

## 2. Run AL Loop (Entropy)
- Код запуска entropy sampling
- Сохранение результатов по итерациям

## 3. Run AL Loop (Random Baseline)
- Код запуска random sampling
- Сохранение результатов

## 4. Visualization
- Построение кривых обучения (accuracy vs n_labeled)
- Оба графика на одном plot

## 5. Analysis
- Расчёт экономии данных
- Таблица сравнения
- Выводы
```

## После каждого шага спрашивай пользователя, хочет ли он продолжить к следующему шагу. Если пользователь не хочет продолжать, дай краткий отчёт о том, что было сделано.

## Требования к реализации:

| Компонент | Требование |
|-----------|------------|
| AL-цикл | Старт с N=50 → 5 итераций → финальная модель |
| Стратегии | Entropy sampling vs Random sampling |
| Визуализация | Кривые обучения на одном графике (accuracy/F1 vs n_labeled) |
| Экономия | Расчёт samples saved при том же качестве vs random baseline |
| Файлы | `agents/al_agent.py` + `notebooks/al_experiment.ipynb` |

## Пример использования:

```bash
# 1. Подготовка данных
.venv/bin/python .opencode/skills/ActiveLearning/scripts/al.py prepare --input data/processed/processed.csv --output data/al_ready

# 2. Запуск AL эксперимента
.venv/bin/python .opencode/skills/ActiveLearning/scripts/al.py run \
    --input data/al_ready/features.npy \
    --labels data/al_ready/labels.npy \
    --n-initial 50 \
    --n-iterations 5 \
    --samples-per-iter 20 \
    --output data/al_results

# 3. Визуализация
.venv/bin/python .opencode/skills/ActiveLearning/scripts/al.py plot \
    --results data/al_results/experiment_results.json \
    --output data/al_results/plots/

# 4. Создание агента
.venv/bin/python .opencode/skills/ActiveLearning/scripts/al.py create-agent \
    --results data/al_results/experiment_results.json \
    --output agents/al_agent.py
```

## Ограничения:
- Для текстовых данных: используй TF-IDF или sentence embeddings как features
- Для изображений: используй предобученный CNN (ResNet) для извлечения features
- Для аудио: используй MFCC или предобученные эмбеддинги
- Сохраняй random state для воспроизводимости
- Тестовый сет должен быть фиксированным на всех итерациях
```

---

**Промпт готов!** Он включает все требуемые компоненты для Трека A (Active Learning):

1. ✅ **AL-цикл**: Старт с N=50 → 5 итераций → финальная модель
2. ✅ **Сравнение стратегий**: Entropy sampling vs Random sampling
3. ✅ **Кривые обучения**: На одном графике (accuracy/F1 vs n_labeled)
4. ✅ **Экономия данных**: Расчёт samples saved при том же качестве
5. ✅ **Файлы**: `agents/al_agent.py` + `notebooks/al_experiment.ipynb`

Агент может быть интегрирован в финальный пайплайн как `active_learning_op`.