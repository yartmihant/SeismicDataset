# 1.7. Сплайновые границы и сетки Fidesys для ECVB

Открытая глава. Пройдены пять пилотов из 30 000 моделей. Сохраняются границы, JOU, материализованный FC, исходный растр, Q2-вход и контрольные суммы. В главе показаны сохранение углов, локальное перестроение и контроль общих индексов источников. Общая доля успеха и выход обучающих карт для всего ECVB ещё не измерены.

## Проверки и рисунки

- [Отчёт о топологии, размерах элементов и перестроениях](docs/dev_1_6_fidesys_mesh.md)
- [Итоговый аудит пяти пилотов](data/dev_1_6_fidesys_mesh/dataset_pilot_checks_v1.json)
- [Сравнение с сеткой по триангуляции](data/dev_1_6_p67s_showcase/fc_mesh_quality_v1/quality.json)
- [Прямоугольные включения №00065](img/dev_1_6_p67s_showcase/cubit_ecvb00065_corners_v2/ECVB00065_corners.png)
- [Качество элементов №00065](img/dev_1_6_p67s_showcase/fc_mesh_quality_v1/scaled_jacobian.png)
- [Поле на более ранней сетке №00065](img/dev_1_6_p67s_showcase/render_v2/ECVB00065/linear/x0700/velocity.mp4)

## Пилотные манифесты

- [ECVB00000](data/dev_1_6_fidesys_mesh/dataset_v1/00000-00999/ECVB00000/manifest.json)
- [ECVB00065](data/dev_1_6_fidesys_mesh/dataset_v1/00000-00999/ECVB00065/manifest.json)
- [ECVB12404](data/dev_1_6_fidesys_mesh/dataset_v1/12000-12999/ECVB12404/manifest.json)
- [ECVB15000](data/dev_1_6_fidesys_mesh/dataset_v1/15000-15999/ECVB15000/manifest.json)
- [ECVB29999](data/dev_1_6_fidesys_mesh/dataset_v1/29000-29999/ECVB29999/manifest.json)

[Дневник](dev_1_7_curved_geometry.py) · [Выполненный ноутбук](dev_1_7_curved_geometry.ipynb)
