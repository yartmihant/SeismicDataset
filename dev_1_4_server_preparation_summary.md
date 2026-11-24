# 1.4. Расчётная область, поглощение и система наблюдений

Как отделить физическое отражение от горизонта от возврата с искусственных границ?

Приняты ядро 1400×700 м, PML350, свободный верх и запись 1 с. Сопоставление ошибки с первым отражением даёт физический масштаб, относительно которого можно оценивать влияние внешней границы.

## Данные и иллюстрации

- [Геометрия и частота, 30 опытов](data/dev_1_5_geometry_frequency/series_v1/summary.json)
- [Сравнение толщины PML и шага времени](data/dev_1_5_geometry_frequency/pml_p4_noabc_tradeoff_v1/analysis/metrics.json)

- [data/dev_1_4_server_preparation/profile_golden_1s_analysis_v1/key_metrics.json](data/dev_1_4_server_preparation/profile_golden_1s_analysis_v1/key_metrics.json)
- [data/dev_1_6_p67s_showcase/proximity_v1/analysis_v2/analysis.json](data/dev_1_6_p67s_showcase/proximity_v1/analysis_v2/analysis.json)
- [img/dev_1_5_geometry_frequency/pml_first_reflection_v1/central_traces.png](img/dev_1_5_geometry_frequency/pml_first_reflection_v1/central_traces.png)
- [img/dev_1_4_server_preparation/production_domain.png](img/dev_1_4_server_preparation/production_domain.png)

## Как выбиралось поглощение

Отделили возврат от профиля массового демпфирования и внешней границы, проверили силу 175-метрового слоя, затем сравнили PML на однородной и двухслойной средах. Усиление тонкого слоя давало поздний рост внутри полосы. Выбор 350 м без ABC учитывает поглощение и допустимый временной шаг.

- [Четыре граничных контроля](docs/dev_1_4_damping_layer_protocol_v1.md)
- [Данные поиска силы профиля](data/dev_1_4_server_preparation/profile_golden_1s_analysis_v1/analysis.json) · [График целевой нормы](img/dev_1_4_server_preparation/damping_search.png)
- [PML в однородной среде](data/dev_1_5_geometry_frequency/pml_homogeneous_saturn_v1/analysis/metrics.json) · [Утончение PML](data/dev_1_5_geometry_frequency/pml_thickness_saturn_v1/analysis/metrics.json) · [Контроль по нескольким отражениям](data/dev_1_5_geometry_frequency/pml_layered_cached_saturn_v1/analysis/metrics.json)
- [Поля внутри PML в позднем окне](img/dev_1_4_server_preparation/pml_tail.png)

[Дневник](dev_1_4_server_preparation.py) · [Выполненный ноутбук](dev_1_4_server_preparation.ipynb)
