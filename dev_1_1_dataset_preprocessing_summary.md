# 1.1. Исходные модели и постановка исследования

Как описана среда и какой физический вопрос возникает при переходе от растровой модели к расчёту волн?

В 60 000 моделях одни и те же свойства заданы разной геометрией горизонтов. Плоские границы можно перенести напрямую, а для ECVB требуется проверить последствия сглаживания.

## Данные и иллюстрации

- [data/dev_1_1_preprocessing/dataset_statistics.json](data/dev_1_1_preprocessing/dataset_statistics.json)
- [src/data_EFVB_out.zip](src/data_EFVB_out.zip)
- [src/data_ECVB_out.zip](src/data_ECVB_out.zip)
- [img/dev_1_1_preprocessing/previews/A/EFVB_overview.png](img/dev_1_1_preprocessing/previews/A/EFVB_overview.png)
- [img/dev_1_1_preprocessing/previews/A/ECVB_overview.png](img/dev_1_1_preprocessing/previews/A/ECVB_overview.png)

## Происхождение и постановка

В оригинальном EFWI физический размер составляет 350 × 350 м. Письмо Василия задаёт 700 × 700 м, а производственный SEM использует ядро 1400 × 700 м. В главе сопоставлены величины, источники, приёмники и отсчёты всех трёх постановок.

- [Исходное письмо](src/vasily_message.txt)
- [Статья EFWI](ref/efwi_feng_2023_main.pdf) · [Численная постановка в приложении](ref/efwi_feng_2023_supplement.pdf) · [Исходный датасет](https://efwi-lanl.github.io/docs/data.html)
- [Сверка методов расчёта](docs/dev_1_4_efwi_comparison_v1.md)

[Дневник](dev_1_1_dataset_preprocessing.py) · [Выполненный ноутбук](dev_1_1_dataset_preprocessing.ipynb)
