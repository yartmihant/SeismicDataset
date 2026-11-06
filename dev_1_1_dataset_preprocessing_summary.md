# 1.1. Предобработка и анализ датасета

Задача — установить формат и физический смысл EFVB/ECVB, проверить исходные поля, подготовить бинарное представление и единые шкалы визуализации. Этап выполнен; новый дневник оформляет существующие результаты без повторного массового расчёта.

## Результат

По 30000 моделей в каждом классе, две компоненты VP/VS, ячейки 70×70 по 10 м, плотность 1000 кг/м³. Строки идут сверху вниз. Материал определяется точной парой (VP, VS), не связной областью.

EFVB: VP 1500–4500 м/с, VS 615,71051–2995,11597 м/с, 3–10 материалов на модель. ECVB: VP 1500–4500 м/с, VS 616,98389–2999,36597 м/с, 3–12 материалов. Пересчёт 5 ноября 2026 года проверил все 120000 полей, комплект ID, форму, конечность, положительность скоростей и объёмного модуля. Гистограммы и контрольные суммы находятся в сохранённом JSON.

Бинарная конвертация использует float64 и повторное точное сравнение записанных NPZ с разобранными полями. PNG используют общие пределы VP/VS всего набора: красный VP, зелёный VS. Изображения не являются расчётными данными.

## Входы и выходы

- Первичная постановка: [src/vasily_message.txt](src/vasily_message.txt).
- Полные входы: `src/data_EFVB_out/vp_*.txt`, `vs_*.txt`, `src/data_ECVB_out/vp_*.txt`, `vs_*.txt`; альтернативно исходные ZIP в `src/`.
- Сохранённая статистика: [data/dev_1_1_preprocessing/dataset_statistics.json](data/dev_1_1_preprocessing/dataset_statistics.json).
- Бинарный manifest: [data/dev_1_1_preprocessing/binary/manifest.json](data/dev_1_1_preprocessing/binary/manifest.json); описанные им 60 NPZ-блоков локально отсутствуют.
- Обзоры: [EFVB](img/dev_1_1_preprocessing/previews/A/EFVB_overview.png), [ECVB](img/dev_1_1_preprocessing/previews/A/ECVB_overview.png), [общая карта шкал](img/dev_1_1_preprocessing/previews/comparison.png).
- Новый график дневника: [гистограмма материалов](img/dev_1_1_preprocessing/dev_1_1_material_count_histogram.png).
- Отчётные представления: [дневник](dev_1_1_dataset_preprocessing.py), [ноутбук](dev_1_1_dataset_preprocessing.ipynb).

## Команды

Короткий отчётный запуск:

```bash
.venv/bin/python dev_1_1_dataset_preprocessing.py
```

Полная обработка — только по отдельному поручению, с новым именем выходного каталога:

```bash
.venv/bin/python -m scripts.dev_1_1_preprocessing.prepare_data convert --source src --output data/dev_1_1_preprocessing/binary_new
.venv/bin/python -m scripts.dev_1_1_preprocessing.dataset_statistics --source src --out data/dev_1_1_preprocessing/dataset_statistics_new.json
```

Исходный `binary/manifest.json` оставлен как историческое свидетельство. Для восстановления отсутствующих NPZ используется новый каталог: завершённый manifest не перезаписывается. Полная конвертация, массовая генерация PNG и пересчёт 60000 моделей при реорганизации не выполнялись.
