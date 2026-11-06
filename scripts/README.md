# Скрипты этапов

Запускать утилиты из корня проекта через `.venv/bin/python -m scripts.<этап>.<модуль>`. Дневники `dev_*.py` остаются императивными; вычислительные CLI в этом каталоге имеют отдельные `main()` и не исполняются при импорте.

## 1.1. Предобработка

- `dev_1_1_preprocessing.prepare_data` — исходные ZIP/тексты → проверенные NPZ, обзорные и массовые PNG.
- `dev_1_1_preprocessing.dataset_statistics` — полный пересчёт диапазонов VP/VS, количества материалов и контрольных сумм из текста или NPZ.

## 1.2. Сглаживание

- `dev_1_2_smoothing.vectorization_probe` — проба общего графа и сплайнов.
- `dev_1_2_smoothing.select_layered_model` — выбор №12404 из бинарного датасета.
- `dev_1_2_smoothing.pilot_mesh`, `pilot_geometry_plot`, `pilot_verify_order`, `pilot_stability` — Q2-геометрия, рисунки и проверки выбранного порядка.
- `dev_1_2_smoothing.pilot_run`, `pilot_windows`, `windows_tensor_probe` — историческая CEED-линия и Windows-инструменты; prefix/DLL задаются явно.
- `dev_1_2_smoothing.pilot_cu_run` — native CU-пара №12404.
- `dev_1_2_smoothing.pilot_compare`, `pilot_cu_compare` — проверка и сопоставление записей; CU-постобработка поддерживает карту перенесённого provenance.
- `dev_1_2_smoothing.model65_geometry`, `model65_mesh`, `model65_geometry_plot`, `model65_stability`, `model65_run`, `model65_compare` — отдельная постановка №65; сеткопостроение требует extra `meshing` и Gmsh.

Общие утилиты: `project_paths.py` — корень, пути этапов, шрифт и проверка исторических хешей; `reporting.py` — типизированное чтение компактных метрик для дневников. Рабочие скрипты не импортируют `research` из TrySEMGPU.

`build_notebooks.py` — конвертация двух дневников штатным py2jupyter с проверкой формата и обратного AST; `--execute` выполняет отчётные ячейки.
