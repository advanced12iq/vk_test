# One-Shot: Является ли логотип на кропе логотипом искомой организации?

## Эпилог
Я немного накосячил когда большие файлы добавлял. Переборщил и ничего не пушилось на гит, поэтому пришлось вернуться на коммит, где этих файлов еще не было и залить все одним коммитом 74966b5. Постараюсь в ближайшее время это разгрести.

## Обзор

Основная идея состоит в использовании простой сверточной нейронной сети (CNN) для создания эмбеддингов как для входного фрагмента изображения, так и для известных логотипов целевой организации. Затем система сравнивает сходство между эмбеддингом фрагмента и эмбеддингами известных логотипов. Если сходство между фрагментом и хотя бы одним из известных логотипов `> threshold`, то фрагмент изображения классифицируется как содержащий логотип целевой организации. CNN обучается с использованием Contrastive Loss.

## Принцип работы

1.  **Генерация Эмбеддингов:** CNN используется для извлечения эмбеддингов признаков как из входного фрагмента изображения, так и из известных логотипов целевой организации.
2.  **Вычисление Сходства:** Вычисляется сходство между эмбеддингом фрагмента и эмбеддингами известных логотипов (cosine_similarity).
3.  **Классификация:** Если максимальное сходство между фрагментом и каким-либо из известных логотипов `> threshold`, то фрагмент изображения классифицируется как содержащий логотип целевой организации.

## Структура Кода

*   **`training_with_tests.ipynb`:** Этот ноутбук содержит обучение, валидацию и тестирование модели. Залил отдельно так как обучал модель на Kaggle. Основные тесты рекомендую смотреть здесь
*   **`inference.ipynb`:** Этот ноутбук предоставляет практический пример того, как использовать обученную модель для инференса на новых фрагментах изображений.

## Проектные Решения

Хотя этот относительно небольшой проект можно было разместить в одном блокноте (как это сделано в `training_with_tests.ipynb`), я организовал это как полноценный проект, чтобы продемонстрировать навыки написания кода, который удобно поддерживать.

## Набор Данных

Первоначально предложенный набор данных оказался непригодным, так как в основном состоял из прямых копий логотипов, а не из реальных изображений с логотипами, встроенными в них.

**Выбранный Набор Данных:** [osld (One Shot Logo Detection)](https://github.com/mubastan/osld)

Этот набор данных предоставляет более реалистичное представление проблемы, состоящее из реальных изображений с логотипами в различных контекстах.

## Использование

**Пожалуйста, обратитесь к блокноту `training_with_tests.ipynb` для проверки тестов** Этот блокнот включает загрузку данных, предварительную обработку, определение модели, обучение, валидацию, тестирование.

## Дальнейшие Улучшения

**Я повышу качество модели в ближайшем времени, но так как это уже будет после дедлайна, пожалуйста, смотрите последний до дедлайна коммит**

*   Эксперименты с архитектурой модели.
*   Эксперименты с различными метриками сходства.
