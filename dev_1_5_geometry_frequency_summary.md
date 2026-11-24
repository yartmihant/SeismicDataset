# 1.5. От эталонного расчёта к массовому

Как сократить стоимость сейсмограммы, сохранив физическую постановку и возможность проверить ответ?

Время четырёх опытов на A100 уменьшилось с 14,67 до 9,78 с. Симметрия EFVB обосновывает четыре прямых источника. Для семи положений в произвольной ECVB предусмотрен пакет p67s.

## Данные и иллюстрации

- [data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json](data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json)
- [data/dev_1_5_geometry_frequency/solver_comparison_v1/analysis.json](data/dev_1_5_geometry_frequency/solver_comparison_v1/analysis.json)
- [img/dev_1_6_p64s_production/comparison_v2/timings.png](img/dev_1_6_p64s_production/comparison_v2/timings.png)

## Этапы оптимизации

- [Первый опыт пакетирования пяти сеток](data/dev_1_4_server_preparation/batch_probe_saturn_v1/full_results.json)
- [Время пяти исходных расчётов](data/dev_1_4_server_preparation/batch_probe_saturn_v1/production_reference_timings.json)
- [Профиль тензорного отклика](data/dev_1_5_geometry_frequency/profile_analysis_v1/comparison.json)
- [Фиксация p5](data/dev_1_5_geometry_frequency/p5_analysis_v1/comparison.json)
- [Узлы демпфирования и специализация p4/p5](data/dev_1_5_geometry_frequency/p45_damping_analysis_v1/comparison.json)
- [Ограничение шага с ABC](data/dev_1_5_geometry_frequency/pml_timing_native_cfl_v1/analysis/metrics.json)
- [PML350 без ABC](data/dev_1_5_geometry_frequency/pml_p4_noabc_tradeoff_v1/analysis/metrics.json)
- [Порядок и размер](data/dev_1_5_geometry_frequency/pml_fixed_orders_v1/analysis/metrics.json)
- [Пакет с полями](data/dev_1_5_geometry_frequency/pml_p6_sources7_v1/analysis/metrics.json)
- [Пакет без полей](data/dev_1_5_geometry_frequency/pml_p6_sources7_surface_v1/analysis/metrics.json)
- [Семь состояний на одной сетке](data/dev_1_5_geometry_frequency/p6_7s_comparison_v2/analysis/metrics.json)
- [p67s и профиль приёмника](data/dev_1_5_geometry_frequency/p67s_receiver_v1/analysis/metrics.json)
- [Стоимость семи возбуждений по этапам](img/dev_1_6_p64s_production/comparison_v2/seven_cost.png)

[Дневник](dev_1_5_geometry_frequency.py) · [Выполненный ноутбук](dev_1_5_geometry_frequency.ipynb)
