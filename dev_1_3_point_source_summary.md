# 1.3. Точность расчёта и выбор размера и порядка элемента

Какой порядок при размере элемента около 10 м сохраняет точность поверхностных записей при приемлемой стоимости?

Для массовой постановки выбран p6 при h≈10 м. На двух пилотах поздняя удалённая разность с уточнённой опорой укладывалась в 0,042%. Ближняя зона точечного источника требует отдельной интерпретации.

## Данные и иллюстрации

- [Сейсмограмма №00065, p5](data/dev_1_3_point_source/central_cu_v1/00065/curved/p5/x350/surface.npz)

- [Сейсмограмма №00065, p4](data/dev_1_3_point_source/central_cu_v1/00065/curved/p4/x350/surface.npz)

- [data/dev_1_3_point_source/central_cu_v1/acceptance_v1/acceptance.json](data/dev_1_3_point_source/central_cu_v1/acceptance_v1/acceptance.json)
- [data/dev_1_3_point_source/central_p3_cu_v1/p3_report_v1/p3_report.json](data/dev_1_3_point_source/central_p3_cu_v1/p3_report_v1/p3_report.json)
- [data/dev_1_4_server_preparation/coarse_order_v1/p5_coarse_analysis_v1/analysis.json](data/dev_1_4_server_preparation/coarse_order_v1/p5_coarse_analysis_v1/analysis.json)
- [img/dev_1_4_server_preparation/coarse_order_v1/p5_coarse_analysis_v1/convergence.png](img/dev_1_4_server_preparation/coarse_order_v1/p5_coarse_analysis_v1/convergence.png)
- [img/dev_1_3_point_source/near_source_traces.png](img/dev_1_3_point_source/near_source_traces.png)

## Две ветви уточнения и независимое сравнение

При 30 Гц на сетке 5 м рабочим стал p4, уточнённым ориентиром — p5. На сетке 10 м приняты p6 и контроль p7. Для Уханова эталоном служит аналитическая трасса Лэмба. Наша оценка PPW построена по минимальной S-скорости и центральной частоте, а точность измерена относительно численной опоры на заданной выборке трасс.

- [Переходы по p на одной Q2-геометрии](data/dev_1_4_server_preparation/coarse_order_v1/analysis_v1/analysis.json)
- [Контроль по p8/p9](data/dev_1_5_geometry_frequency/pml_fixed_orders_v1/analysis/metrics.json)
- [Работа Е. М. Уханова](ref/Uhanov_2025_Lamb.pdf) · [Рисунок 33 и таблица со с. 48](img/dev_1_3_point_source/uhanov_fig33_table.png)

[Дневник](dev_1_3_point_source.py) · [Выполненный ноутбук](dev_1_3_point_source.ipynb)
