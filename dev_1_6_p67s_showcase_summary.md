# 1.6. Расчёт EFVB и формат датасета для инверсии

Контракт одного ID включает сейсмограммы, ядро Q2 и два поля Vp/Vs размером 140 × 70. Текущие четыре прямые записи дают семь по симметрии. Для всех тринадцати положений EFVB нужны ещё три прямых опыта. На EFVB00000 проверены все 9800 ячеек восстановленных карт. Для 30 000 EFVB оценка полного выпуска с 13 записями составляет 885,6 ГБ, для 30 000 ECVB с дополнительной геометрией — 956,7 ГБ. Объёмы рассчитаны при FP64. Геометрия оценена по одному плоскому и пяти криволинейным пилотам.

## Постановка, входы и состояние

- [Производственная спецификация](docs/dev_1_6_p64s_production_spec.txt)
- [Чтение исходных пар](src/data_EFVB_out.zip)
- [План и подтверждение завершения первого чанка](data/dev_1_6_p64s_production/chunks02_04_v1/plan.json)
- [Снимок 24 ноября, 01:04 МСК](data/dev_1_6_p64s_production/chunks02_04_v1/launch_snapshot.json)
- [Объёмы дополнительных артефактов ECVB](data/dev_1_6_fidesys_mesh/dataset_pilot_checks_v1.json)

## Формат на примере EFVB00000

- [Q2-ядро и материал](data/dev_1_6_p64s_production/format_pilot_v1/EFVB00000/sem_input.npz)
- [Карты скоростей и расстановка источников](img/dev_1_6_p64s_production/EFVB00000_training_grid.png) · [Проверка симметрии](data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json)
- [Цель инверсии в центрах ячеек](data/dev_1_6_p64s_production/format_pilot_v1/EFVB00000/material_140x70.npz)

[Дневник](dev_1_6_p67s_showcase.py) · [Выполненный ноутбук](dev_1_6_p67s_showcase.ipynb)
