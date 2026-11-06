# 1.2. Пробный расчёт со сглаживанием

Задача — сопоставить исходную ячеистую среду и сглаженные границы на согласованной Q2-геометрии, оценив изменение сейсмограмм и полей относительно доступных контролей дискретизации. Этап выполнен в объёме двух пилотов; это не расчёт всего датасета и не независимая валидация solver API.

## Постановка и результаты

ECVB 12404: четыре широких слоя, 4900 элементов в обеих средах, центральная вертикальная сила Рикера 15 Гц, свободный верх, ABC по бокам и снизу. Историческая Windows CEED-серия включает p2–p5 и временной контроль p4. Linux CU-повтор выполнен на p5, 123201 узле, с шагом 1/36 мс до 2 с. Приёмники: 71 точка, 2001 отсчёт Ux/Uy включая t=0; поля — в четырёх общих моментах и физических точках.

Разность Ux/Uy: полная запись — 0,83958%/0,50901%; поздняя кода 1–2 с — 18,47970%/16,08199%; поле при 2 с — 13,94981%/14,55042%. В исторической серии эффект поздней коды превышает сумму пространственных поправок в 109,60/65,01 раза. Для полной записи отношение 2,76/2,40 не достигает протокольного порога 5. CU-повтор согласуется с сохранёнными агрегатами, но старые полные Windows-массивы здесь отсутствуют.

ECVB 00065: пять материалов, восемь связных областей, p4, 4900/6075 элементов, 72 шага на 1 мс. Изменено около 1,04% площади материала при сохранённой топологии. Полная запись отличается на 2,23%/1,37%, поздняя кода — на 13,75%/6,23%, поле при 2 с — на 26,07%/22,02%. Отдельной серии сходимости №65 нет, поэтому геометрический эффект не отделён полностью от изменения пространственной дискретизации.

Исходное письмо задаёт перемещение и пять источников; пилот использует силу и один центральный источник. Код процесса 143 у исходного CU-прогона после сохранения полной записи указан в историческом отчёте отдельно от успешного терминального solver-результата.

## Входы и выходы

- Входные поля: `src/data_ECVB_out/`; архивная геометрия №12404 — `data/dev_1_2_smoothing/pilot_geometry/selected_model.npz`, `frozen_q2_geometry.npz`; №65 — `data/dev_1_2_smoothing/model65/geometry/model65.npz`. Три малых NPZ сохранены локально, но исключены из нового Git вместе с массивами.
- [Протокол №12404](docs/dev_1_2_smoothing/pilot_geometry/PROTOCOL.md), [отчёт исторической серии](docs/dev_1_2_smoothing/PILOT_REPORT.md), [CU-повтор](docs/dev_1_2_smoothing/pilot_cu/REPORT.md).
- [Протокол №65](docs/dev_1_2_smoothing/model65/PROTOCOL.md), [отчёт №65](docs/dev_1_2_smoothing/model65/REPORT.md).
- Метрики: [CU №12404](data/dev_1_2_smoothing/pilot_cu/comparison/report.json), [историческая сходимость №12404](data/dev_1_2_smoothing/pilot_runs/comparison/metrics.json), [№65](data/dev_1_2_smoothing/model65/comparison/report.json).
- Локальные полные CU-массивы: `data/dev_1_2_smoothing/pilot_cu/{raster,curved}_p5_s36/{surface,fields}.npz`.
- Иллюстрации: `img/dev_1_2_smoothing/{vectorization_trial,pilot_geometry,pilot_runs,pilot_cu,model65}/`.
- [Презентация](presentation/dev_1_2_seismic_dataset_sem_review_20261105.pptx), [дневник](dev_1_2_smoothing_pilot.py), [ноутбук](dev_1_2_smoothing_pilot.ipynb).

## Команды

Отчёт без расчёта:

```bash
.venv/bin/python dev_1_2_smoothing_pilot.py
```

Адресный повтор CU после подготовки native-профиля, только по отдельному поручению:

```bash
.venv/bin/python -m scripts.dev_1_2_smoothing.pilot_cu_run --kind raster --order 5 --substeps 36 --out data/dev_1_2_smoothing/pilot_cu_new/raster_p5_s36
.venv/bin/python -m scripts.dev_1_2_smoothing.pilot_cu_run --kind curved --order 5 --substeps 36 --out data/dev_1_2_smoothing/pilot_cu_new/curved_p5_s36
.venv/bin/python -m scripts.dev_1_2_smoothing.pilot_cu_compare --root data/dev_1_2_smoothing/pilot_cu_new --out data/dev_1_2_smoothing/pilot_cu_new/comparison
```

Рисунки новых сравнений сохраняются в зеркальном каталоге `img/dev_1_2_smoothing/`. Старые результаты не перезаписываются. Новые расчёты при оформлении проекта не выполнялись.
