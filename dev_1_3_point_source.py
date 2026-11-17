""" # SeismicDataset. Часть 1: подготовка и исследование моделей # """

""" ## Глава 1.3. Точечные источники 30 Гц и демпфирующее окружение ## """

"""
### Задача и граница отчёта

Исследовать ECVB 12404 и ECVB 00065 с точечными источниками Рикера 30 Гц, новой сеткой и демпфирующим окружением. Серия p4/p5 ограничена центральным источником x=350 м: восемь основных опытов и четыре временных контроля. По дополнительному поручению проверяется p3 на тех же четырёх геометриях с четырьмя собственными временными контролями. Остальные четыре положения источника отложены пользователем.

Расширенные сетки построены, запускатель и обработка реализованы, короткие native-проверки выполнены. Дневник только читает выбранные сохранённые JSON и несколько готовых PNG: не строит сетки, не загружает тяжёлые массивы, не запускает решатель и не создаёт анимации. Число завершённых опытов определяется по метаданным при каждом отчётном запуске; старт процесса не считается результатом.
"""

from pathlib import Path
from typing import Mapping

from IPython.display import Image, display

from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence


data_root = PROJECT_ROOT / "data/dev_1_3_point_source"
image_root = PROJECT_ROOT / "img/dev_1_3_point_source"


""" ## 1. Согласованная постановка ## """

"""
Верх основной области содержит 140 прямых рёбер по 5 м. Все пять предусмотренных координат источника и 141 приёмник x=0,5,…,700 м являются вершинами при p3, p4 и p5. Одинаковой остаётся результирующая сила, а не введённая энергия. В текущих расчётах возбуждается только центральный узел.

Основная область [0,700]×[−700,0] м, полная область со слоем 175 м — [−175,875]×[−875,0] м. Ось y направлена вверх, поверхность y=0 свободна на всём протяжении. ABC находится только на внешних боках и дне. Материал продолжается горизонтально по бокам, вертикально снизу, от соответствующего угла в нижних углах.
"""

preflight = read_json(data_root / "preflight.json")
decisions = mapping(preflight["accepted_decisions"])
print("Верх:", decisions["boundary_elements"], "элементов по", decisions["boundary_element_size_m"], "м")
print("Порядки исходного preflight:", decisions["solution_orders"], "; дополнительный контроль: p3; одинаковая амплитуда:", decisions["source_peak_force_n_per_m"], "Н/м")


r"""
Рикер имеет частоту 30 Гц и центр t0=0,05 с. Главный пик направлен вниз. Единственному узлу передаётся сила без множителя площади и без деления на массу:

$$
a=\pi\,30\,(t-0.05),\qquad F_y(t)=-F_0(1-2a^2)e^{-a^2},\qquad F_0=1\ \mathrm{Н/м}.
$$

Работа является отдельной диагностикой, не нормировкой амплитуды. Она интегрируется по фактическим силе и скорости источникового узла, включая отрицательную мощность:

$$
W(t)=\int_0^t F_y(\tau)v_y(x_s,0,\tau)\,d\tau.
$$

Канал EXTERNAL_LINEAR_WORK в semgpu2 возвращает u·F и не подменяет этот интеграл.
"""

"""
Начальное состояние серии — U=V=A=0. В native CU начальный force response тоже нулевой; первая сборка динамической силы выполняется при t=Δt. Поэтому малый аналитический хвост Рикера в t=0, равный 9,84949·10⁻⁹ Н/м, присутствует во входной таблице, но не задаёт начальное ускорение. Это общее для всех запусков начальное соглашение, а не только особенность записи. Для всех положительных времён сохранённая сила совпадает с заданным Рикером с максимальным абсолютным отклонением 1,56·10⁻¹⁵ Н/м; главный пик равен −1 Н/м при t=0,05 с. Порядок начального события и первого шага разобран в SOLVER_AUDIT.md; исторические массивы не исправлялись.
"""

""" ## 2. Построенные сетки и проверенная геометрия ## """

"""
Созданы четыре замороженные Q2-сетки. Для регулярных вариантов и curved №12404 полная область содержит 36750 элементов. Новая curved-сетка №65 построена отдельно с закреплённым равномерным верхом и содержит 42692 элемента. Для обоих порядков используется одна и та же геометрия каждого варианта; при повышении p сплайны заново не подгоняются.

Сохранённый аудит проверяет конформность, площадь полной области, положительность якобиана на всей Q2-ячейке через коэффициенты Бернштейна, верх 140×5 м и все источники/приёмники в восьми GLL-вложениях. Для №65 проверены связность восьми регионов, одинаковый набор соседств raster/curved и короткий контакт материалов длиной 10 м. Это геометрический сертификат, не независимая проверка решателя.
"""

geometry = read_json(data_root / "meshes_v1/validation.json")
print("Геометрический аудит:", geometry["status"])
for value in sequence(geometry["meshes"]):
    mesh = mapping(value)
    print(f"ECVB {mesh['model_id']} {mesh['kind']}: {mesh['elements']} элементов; min detJ ≥ {number(mesh['det_jacobian_certified_min_m2']):.6f} м²")
    for embedding_value in sequence(mesh["embeddings"]):
        embedding = mapping(embedding_value)
        print(f"  p{embedding['p']}: {embedding['global_nodes']} узлов; источники {embedding['sources']}/5, приёмники {embedding['receivers']}/141")
print("Соседства регионов №65 совпадают:", geometry["model65_same_region_adjacency"])
display(Image(filename=str(image_root / "meshes_v1/meshes.png"), width=1100))


"""
Все четыре сетки и восемь вложений прошли сохранённые проверки. Сертифицированная нижняя граница якобиана новой curved-сетки №65 около 0,962 м² положительна, но её меньшие элементы требуют более короткого временного шага, чем сетки №12404.
"""

""" ## 3. Слой и начальная оценка шага ## """

r"""
Массовое демпфирование имеет соглашение C=αM. В основной области α=0; в нижних углах вклады направлений складываются:

$$
\alpha(x,y)=\alpha_{\max}\left[\left(\frac{\max(-x,x-700,0)}{175}\right)^2+\left(\frac{\max(-y-700,0)}{175}\right)^2\right].
$$

Для серии выбрана начальная оценка αmax≈346,177 с⁻¹. Она получена по максимальному VP обеих моделей и желаемому затуханию при двойном проходе; фактический коэффициент отражения ещё не проверен. Слой не является PML. Длины волн при 30 Гц и геометрический размер слоя не доказывают эффективность для всего спектра Рикера и всех углов падения.

Восемь оценок устойчивости используют фактические GLL-массы, верхнюю границу жёсткости и вклады αM/ABC. Для №12404 выбран общий K=64 внутренних шага на 1 мс; для №65 — K=224. Временные контроли p4 используют 2K. Это консервативный старт, а точность времени проверяется по расчётам.
"""

for stability_path in sorted((data_root / "meshes_v1").glob("*_stability.json")):
    stability = read_json(stability_path)
    print(stability_path.name, "— минимум подшагов/мс:", stability["substeps_per_ms"], "; начальный предел Δt, с:", stability["dt_start_s"])


""" ## 4. Выполненные проверки native и записи ## """

"""
В собственной среде установлена совместная сборка native CU и CEED, GPU доступен. Синтетические короткие CEED-опыты до 0,06 с выполнены для p4/p5 на CPU и GPU `/gpu/cuda/gen`: проверены точечная сила, U/V, положительная энергия и включение обоих концов временной записи. Они не являются расчётами двух научных моделей и не заменяют их сходимости.

Разность CPU/GPU записей U/V находится на уровне примерно 2·10⁻¹⁴ по относительной L2-норме. Формат compact дополнительно проверен на CPU: общие сохранённые массивы побитово совпали с full. На полной сетке №12404 p5 выполнено сравнение native CU с CEED `/gpu/cuda/gen` до t=0,01 с: CU быстрее в 1,908 раза по полному времени, включая подготовку, запись и сжатие. Относительные различия поверхности — 2,70·10⁻¹⁶ для U и 2,61·10⁻¹⁵ для V. Это короткий опыт до пика Рикера t=0,05 с: он не доказывает двухсекундного согласования или сходимости, а измеренное ускорение не является чистым временем ядра.
"""

runtime = read_json(data_root / "runtime_combined_v1/profile.json")
print("Совместная сборка native: runtime_combined_v1/profile.json; точные пути, версии и SHA256 сохранены в профиле и метаданных опытов.")
parity = read_json(data_root / "smoke_gpu_v1/cpu_gpu_parity.json")
for order, value in parity.items():
    print("CPU/GPU synthetic p" + order + ":", mapping(value))
benchmark = read_json(data_root / "benchmark_v1/12404_raster_p5/metadata.json")
print("Короткий CEED full-mesh тест:", benchmark["status"], mapping(benchmark["runtime"])["current_time_s"], "с")
cu_benchmark = read_json(data_root / "benchmark_cu_v1/comparison.json")
print("CU/CEED benchmark:", cu_benchmark["status"], "; ускорение полного времени:", cu_benchmark["wall_speedup"])
print("Ограничения короткого сравнения:", cu_benchmark["limitations"])


""" ## 5. Запускатель, регистрация и постобработка ## """

"""
Запускатель реализует отдельный источник, αM, ABC, проверку свидетельства устойчивости до старта, неизменяемый каталог результата и фиксацию фактических native SHA256. Batch связывает сетку с её аудитом и запускает последовательность отдельных процессов; неполные каталоги автоматически не перезаписываются.

Для всех 141 приёмника сохраняются Ux, Uy, Vx, Vy в 2001 момент 0…2 с через 1 мс. Работа источника вычисляется по внутренним шагам. Поля основных тестовых запусков пишутся через 0,01 с, включая все узлы окружения; 201 кадр на двухсекундную запись. Compact сохраняет Vx/Vy и кинетическую/упругую плотности энергии в float64 без округления. Всеузловые перемещения и избыточные узловые энергии отсутствуют в compact; суммы энергии всё равно вычисляются из соответствующих native-каналов и сохраняются отдельно.

Каналы узловой энергии уже собраны из соседних элементов, поэтому их суммируют один раз по глобальным узлам. Строгий баланс по основной области/слою и накопленные потери ABC/демпфирования пока не реализованы; сумма энергии и работа источника не объявляются полным замкнутым балансом.

Compare обрабатывает только завершённые запуски и строит четыре компоненты сейсмограмм и нормы p/dt/геометрической разности в общих приёмниках. Animate читает завершённые поля, создаёт ролики скорости частиц и полной плотности энергии с постоянными шкалами. Разность полей на разных сетках напрямую не вычисляется. Наличие этих утилит не означает, что требуемые физические результаты уже получены.
"""

sampling = mapping(preflight["sampling"])
print("Сейсмограмма:", sampling["seismogram_count"], "×", sampling["receiver_count"], "×", sampling["requested_components"])
print("Поле:", sampling["test_field_frame_count"], "кадр через", sampling["test_field_interval_s"], "с")


""" ## 6. Центральная серия: двенадцать запусков ## """

"""
Исходная серия p4/p5 `central_cu_v1` содержит восемь основных двухсекундных опытов native CU для x=350 м и четыре временных контроля p4 с удвоенным K. Поля предусмотрены для всех восьми основных p4/p5-опытов; временные контроли сохраняют поверхность и историю источника. Успех каждого расчёта подтверждается только итоговыми метаданными `completed`. Стартовые метаданные сами по себе не доказывают, что процесс сейчас работает. Дополнительная серия p3 рассмотрена в разделе 12.
"""

central = data_root / "central_cu_v1"
plan = read_json(central / "batch_manifest.json")
completed_main = 0
completed_controls = 0
started_without_result = 0
failed = 0
for value in sequence(plan["jobs"]):
    job = mapping(value)
    output = Path(str(job["output"]))
    metadata_path = output / "metadata.json"
    job_status = "ожидается"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") == "completed":
            if job["phase"] == "main":
                completed_main += 1
            else:
                completed_controls += 1
            counts = mapping(metadata["recorded_counts"])
            job_status = f"завершён; записи {counts}"
        else:
            failed += 1
            job_status = f"не завершён: {metadata.get('status')}"
    elif (output / "metadata.failed.json").is_file():
        failed += 1
        job_status = "сохранена ошибка"
    elif (output / "metadata.started.json").is_file():
        started_without_result += 1
        job_status = "есть стартовые метаданные, итог ещё не сохранён"
    print(job["job_id"], "—", job_status)
print(f"Завершено: основные {completed_main}/8; временные контроли {completed_controls}/4; ошибки {failed}.")
print(f"План до сжатия: {number(plan['total_uncompressed_upper_bound_bytes']) / 1024**3:.2f} GiB; K: {plan['common_substeps_per_ms']}.")


""" ## 7. Полная линия приёмников: p- и временные поправки ## """

"""
Первый обязательный результат — запись всех 141 приёмника, включая узел приложения силы. Норма вычисляется по общим отсчётам времени и физическим приёмникам без весов. p-поправка нормирована на p5; временная поправка — на p4 с половинным шагом. Исключение окрестности источника меняет оцениваемую величину и не может заменить результат по полной линии.

Основной источник данных этого раздела — `comparison_all_v1/report.json`. До его появления используется только явно обозначенный первый сохранённый отчёт №12404 raster; он не выдаётся за результат всей серии. На рисунках показываются выбранные пары raster, а численные итоги охватывают все завершённые p-пары в выбранном отчёте.
"""


def percent(value: object) -> str:
    """Отформатировать относительную норму, сохраняя отсутствие определённого значения."""
    return "не определено" if value is None else f"{100 * number(value):.5g}%"


def component_percentages(components: Mapping[str, object], key: str = "relative_l2") -> str:
    """Собрать четыре значения в фиксированном порядке компонент."""
    return "; ".join(f"{name}={percent(mapping(components[name]).get(key))}" for name in ("Ux", "Uy", "Vx", "Vy"))


comparison_path = central / "comparison_all_v1/report.json"
final_comparison_ready = comparison_path.is_file()
if not final_comparison_ready:
    comparison_path = central / "comparison_12404_raster_v1/report.json"
comparison_report: dict[str, object] | None = read_json(comparison_path) if comparison_path.is_file() else None
p_comparisons: list[dict[str, object]] = []
surface_comparisons: list[dict[str, object]] = []
all_receiver_over_one_percent: list[str] = []
if comparison_report is None:
    print("Сохранённых p-сопоставлений пока нет.")
else:
    print("Итоговый отчёт серии:" if final_comparison_ready else "Промежуточный отчёт одной пары:", comparison_path.relative_to(PROJECT_ROOT))
    print("Завершённых входных опытов:", comparison_report["completed_run_count"])
    for value in sequence(comparison_report["comparisons"]):
        comparison = mapping(value)
        if comparison["comparison"] not in ("p", "dt"):
            continue
        surface_comparisons.append(comparison)
        if comparison["comparison"] == "p":
            p_comparisons.append(comparison)
        for window_value in sequence(comparison["windows"]):
            window = mapping(window_value)
            if window["time_window_s"] != [0.0, 2.0] or window["excluded_radius_m"] != 0.0:
                continue
            components = mapping(window["components"])
            print(comparison["comparison"], comparison["value"], "→", comparison["reference"], "; все 141 приёмник:", component_percentages(components))
            for name in ("Ux", "Uy", "Vx", "Vy"):
                relative = mapping(components[name])["relative_l2"]
                if comparison["comparison"] == "p" and relative is not None and number(relative) > 0.01:
                    all_receiver_over_one_percent.append(f"{comparison['value']} {name}: {percent(relative)}")
    selected_images = sequence(comparison_report.get("images", []))
    for model_id, substeps in ((12404, 64), (65, 224)):
        filename = f"p_m{model_id:05d}_raster_p4_x350_k{substeps}_minus_m{model_id:05d}_raster_p5_x350_k{substeps}_full.png"
        for image_value in selected_images:
            if isinstance(image_value, str) and Path(image_value).name == filename:
                saved_image = PROJECT_ROOT / image_value
                if saved_image.is_file():
                    print(f"ECVB {model_id}: полная сейсмограмма, источник включён.")
                    display(Image(filename=str(saved_image), width=1000))


"""
Разность у источника не скрывается общей оценкой вдали от него. Например, сохранённая первая пара №12404 raster даёт около 4% для Uy и Vy по всей линии; конкретные значения выводятся из отчёта выше. Поэтому малые нормы вне 20 м нельзя приписать всей линии, включающей источник. Причину и пространственное распределение разности рассматривает отдельная диагностика ниже; теоретическая сходимость точечного значения здесь не доказана.
"""

""" ## 8. Удалённые приёмники far20 и временной контроль ## """

r"""
Дополнительная область оценки far20 включает только приёмники с расстоянием не менее 20 м от источника. Для полной записи 0…2 с используются три рабочих критерия протокола, а не допуск, заданный пользователем:

$$
\frac{\|q_{p4}-q_{p5}\|_2}{\|q_{p5}\|_2}\leq0.01,\qquad
\frac{\|q_{p4,\Delta t}-q_{p4,\Delta t/2}\|_2}{\|q_{p4,\Delta t/2}\|_2}\leq0.001,\qquad
\|q_{p4,\Delta t}-q_{p4,\Delta t/2}\|_2\leq\frac{\|q_{p4}-q_{p5}\|_2}{5}.
$$

Последний критерий сравнивает абсолютные нормы поправок, а не их относительные проценты с разными знаменателями. Все четыре компоненты проверяются отдельно. Окно 1…2 с служит дополнительной диагностикой и не заменяет полную запись при оценке критериев. Если временного контроля нет, результат остаётся незавершённым независимо от малости p-поправки.
"""

for comparison in surface_comparisons:
    for window_value in sequence(comparison["windows"]):
        window = mapping(window_value)
        if window["time_window_s"] == [0.0, 2.0] and window["excluded_radius_m"] == 20.0:
            print(comparison["comparison"], comparison["value"], "→", comparison["reference"], "; far20:", component_percentages(mapping(window["components"])))

acceptance_path = central / "acceptance_v1/acceptance.json"
acceptance_report: dict[str, object] | None = read_json(acceptance_path) if acceptance_path.is_file() else None
far20_pass: bool | None = None
if acceptance_report is None:
    print("Итоговый отчёт рабочих критериев ещё не сохранён; временной запас не подтверждён.")
else:
    print("Рабочие критерии far20:", acceptance_report["status"])
    pass_value = acceptance_report.get("all_components_pass")
    far20_pass = pass_value if isinstance(pass_value, bool) else None
    for value in sequence(acceptance_report["cases"]):
        case = mapping(value)
        print(f"ECVB {case['model_id']} {case['kind']}: {case['status']}")
        if "full_far20" not in case:
            print("  Отсутствуют:", case.get("missing_runs"))
            continue
        for name, component_value in mapping(mapping(case["full_far20"])["components"]).items():
            component = mapping(component_value)
            ratio_value = component.get("dt_to_p_absolute_ratio")
            ratio = "не определено" if ratio_value is None else f"{number(ratio_value):.5g}"
            print(f"  {name}: p={percent(component['p_relative_l2'])}; dt={percent(component['dt_relative_l2'])}; абсолютное dt/p={ratio}; все три критерия={component['accepted']}")
    for value in sequence(acceptance_report["geometry_attribution"]):
        attribution = mapping(value)
        print(f"ECVB {attribution['model_id']}: отделимость геометрической разности — {attribution['status']}")
        for window_value in sequence(attribution.get("windows", [])):
            window = mapping(window_value)
            if window["time_window_s"] != [0.0, 2.0]:
                continue
            for name, component_value in mapping(window["components"]).items():
                component = mapping(component_value)
                ratio_value = component["effect_over_correction_sum"]
                ratio = "не определено" if ratio_value is None else f"{number(ratio_value):.5g}"
                print(f"  {name}: геометрическая разность / сумма четырёх p/dt-поправок={ratio}; запас ×5={component['five_times_margin']}")

if final_comparison_ready and comparison_report is not None:
    for model_id, substeps in ((12404, 64), (65, 224)):
        filename = f"p_m{model_id:05d}_raster_p4_x350_k{substeps}_minus_m{model_id:05d}_raster_p5_x350_k{substeps}_full_far20.png"
        for image_value in sequence(comparison_report.get("images", [])):
            if isinstance(image_value, str) and Path(image_value).name == filename:
                saved_image = PROJECT_ROOT / image_value
                if saved_image.is_file():
                    print(f"ECVB {model_id}: дополнительная сейсмограмма far20; полная линия приведена отдельно.")
                    display(Image(filename=str(saved_image), width=1000))
else:
    far_plot_path = central / "comparison_12404_raster_far20_v1/report.json"
    if far_plot_path.is_file():
        far_plot = read_json(far_plot_path)
        saved_image = PROJECT_ROOT / str(far_plot["image"])
        if saved_image.is_file():
            print("Первая пара №12404 raster: дополнительный рисунок far20.")
            display(Image(filename=str(saved_image), width=1000))


"""
Даже выполнение всех трёх критериев far20 относится только к указанным приёмникам и центральному источнику. Наблюдаемые p/dt-поправки не являются строгими оценками ошибки; отношение геометрической разности к их сумме — исследовательская проверка запаса, а не доказательство причинности. Контроль отражения от слоя и независимая валидация решателя остаются отдельными задачами.
"""

""" ## 9. Узел источника: четыре отдельные диагностики ## """

"""
Для каждой пары raster/curved обеих моделей отчёт показывает долю узла источника в квадрате общей p-разности, относительную разность самого источникового сигнала и остаток после исключения только одного узла. Эта диагностика отличается от far20: она не исключает соседние приёмники. Дополнительно проверяется одинаковая история приложенной силы p4/p5; амплитуда не калибруется по энергии.
"""

source_diagnostic_count = 0
for model_id in (12404, 65):
    for kind in ("raster", "curved"):
        diagnostic_path = central / f"source_diagnostic_{model_id:05d}_{kind}_v1/point_source_diagnostic.json"
        if not diagnostic_path.is_file():
            print(f"ECVB {model_id} {kind}: диагностика источника ожидается.")
            continue
        diagnostic = read_json(diagnostic_path)
        if diagnostic.get("status") != "completed_source_localization":
            print(f"ECVB {model_id} {kind}: {diagnostic.get('status')}")
            continue
        source_diagnostic_count += 1
        force = mapping(diagnostic["source_force"])
        print(f"ECVB {model_id} {kind}: одинаковая история силы={force['identical_history']}; max |ΔFy|={force['maximum_abs_difference_n_per_m']} Н/м")
        for name, component_value in mapping(mapping(diagnostic["surface"])["components"]).items():
            component = mapping(component_value)
            print(f"  {name}: доля источника в квадрате разности={percent(component['source_share_of_squared_difference'])}; относительная разность в источнике={percent(component['source_only_relative_l2'])}; без одного узла={percent(component['excluding_only_source_relative_l2'])}")
print(f"Завершённых диагностик источника: {source_diagnostic_count}/4.")


"""
Локализация наблюдаемой разности в одном узле — эмпирический результат. Она не доказывает теоретическую причину разности и не позволяет объявить сходимость самого значения в узле приложения точечной силы.
"""

""" ## 10. Анимации скорости частиц и плотности энергии ## """

"""
Для каждой модели предусмотрены два ролика по 201 кадру: модуль скорости частиц и сумма кинетической/упругой плотностей энергии. Четыре панели сопоставляют raster p4/p5 и curved p4/p5. В отчётном дневнике читаются только описание роликов и один готовый кадр скорости на модель; предпочтителен кадр t=0,30 с из `preview_frames.json`, где фронты лучше видны. Если он ещё не сохранён, показывается финальный кадр t=2 с. Сами поля и видео заново не обрабатываются.

Картинка использует ближайшие глобальные узлы на общей сетке пикселей; это пространственная выборка для визуализации. Все исходные узловые поля сохранены отдельно. Шкалы постоянны по времени и общие для четырёх панелей; логарифмическое отображение, если включено, не является временным усилением. Узловая плотность энергии у границ материалов является восстановленной величиной решателя.
"""

animation_count = 0
animation_preview: Path | None = None
for model_id in (12404, 65):
    animation_path = central / f"animation_{model_id:05d}_v1/animation.json"
    if not animation_path.is_file():
        print(f"ECVB {model_id}: итоговые анимации ожидаются.")
        continue
    animation = read_json(animation_path)
    if animation.get("status") != "animation_complete":
        print(f"ECVB {model_id}: {animation.get('status')}")
        continue
    animation_count += 1
    print(f"ECVB {model_id}: {animation['frame_count']} кадров, интервал {animation['physical_interval_s']} с, конец {animation['end_s']} с; шкала {animation['color_scale']}.")
    animation_preview = None
    preview_time_s = number(animation["end_s"])
    for output_value in sequence(animation["outputs"]):
        if not isinstance(output_value, str):
            continue
        artifact = PROJECT_ROOT / output_value
        if artifact.suffix == ".mp4":
            print("Ролик:", output_value)
        elif artifact.name == "speed_final.png" and artifact.is_file():
            animation_preview = artifact
    preview_path = animation_path.parent / "preview_frames.json"
    if preview_path.is_file():
        previews = read_json(preview_path)
        if previews.get("status") == "extracted_from_completed_movies":
            for frame_value in sequence(previews["frames"]):
                frame = mapping(frame_value)
                preview_image = PROJECT_ROOT / str(frame["image"])
                if Path(str(frame["movie"])).name == "speed.mp4" and frame["frame_index"] == 30 and frame["physical_time_s"] == 0.3 and preview_image.name == "speed_t0_30.png" and preview_image.is_file():
                    animation_preview = preview_image
                    preview_time_s = number(frame["physical_time_s"])
                    break
    if animation_preview is not None:
        print(f"Кадр скорости ECVB {model_id}: физическое время {preview_time_s:.2f} с.")
        display(Image(filename=str(animation_preview), width=1000))
print(f"Готовых комплектов анимаций: {animation_count}/2.")


""" ## 11. Отдельное двухсекундное сравнение CU и CEED ## """

"""
Согласование backend проверено отдельно на №12404 raster p4 с K=64 и центральным источником до 2 с. Завершённый CEED-опыт не подставляется в CU-пары p4/p5 или временные контроли. Это проверка согласования реализаций, не независимая валидация физического решения.
"""

backend_path = data_root / "backend_comparison_2s_v1/report.json"
if backend_path.is_file():
    backend_report = read_json(backend_path)
    print("CU/CEED, 2 с:", backend_report["status"])
    for name in ("displacement_m", "velocity_m_s", "kinetic_j_per_m", "elastic_j_per_m", "source_work_j_per_m"):
        norm = mapping(mapping(backend_report["norms"])[name])
        print(f"  {name}: относительная L2={number(norm['relative_l2']):.3e}; max |Δ|={number(norm['max_absolute']):.3e}")
else:
    print("Двухсекундное сравнение backend пока не сохранено.")
backend_fields_path = data_root / "backend_parity_12404_raster_p4_v1.json"
if backend_fields_path.is_file():
    backend_fields = read_json(backend_fields_path)
    for value in sequence(backend_fields["fields"]):
        field = mapping(value)
        velocity = mapping(field["velocity_m_s"])
        print(f"  Поле скорости, t={field['time_s']} с: относительная L2={number(velocity['relative_l2']):.3e}; max |ΔV|={number(velocity['max_abs_difference']):.3e} м/с")


""" ## 12. Дополнительная проверка p3 относительно p4 и p5 ## """

"""
Новая серия `central_p3_cu_v1` добавляет четыре p3-опыта и четыре контроля с половинным шагом времени. Геометрические карты Q2, материалы, источник и начальное соглашение сохранены. Основные шаги совпадают с p4/p5: K=64 для №12404 и K=224 для №65; контролям p3 соответствуют K=128 и K=448. Предусмотрены 2001 поверхностный отсчёт и 201 полевой кадр основных опытов. Снимки прежних утилит находятся в `provenance_pre_p3_v1`; прежние расчётные массивы и отчёт p4/p5 не переписываются.

Сравнения p3/p4 и p3/p5 рассматриваются отдельно: знаменатель p-нормы — соответствующее опорное решение. Временная поправка относится к p3 и нормирована на p3 с половинным шагом. Рабочие ориентиры полного окна far20 остаются 1%, 0,1% и абсолютное dt/p≤0,2. Полная линия с источником, позднее окно и локализация ошибки выводятся отдельно; прохождение far20 не означает сходимости точечного значения.

Условие dt/p≤0,2 выполнить легче при большей пространственной поправке. Поэтому возможное прохождение p3 при непройденном временном запасе p4 не означает большей точности p3. Для выбора порядка нужно сопоставлять величины поправок и отдельный запас геометрического эффекта.
"""

p3_root = data_root / "central_p3_cu_v1"
p3_preflight_path = data_root / "p3_preflight_v1/validation.json"
if p3_preflight_path.is_file():
    p3_preflight = read_json(p3_preflight_path)
    print("Предварительный аудит p3:", p3_preflight.get("status"))
p3_report_path = p3_root / "p3_report_v1/p3_report.json"
p3_report = read_json(p3_report_path) if p3_report_path.is_file() else None
p3_complete = p3_report is not None and p3_report.get("is_final_complete") is True and p3_report.get("completed_input_runs") == 16
p3_geometry_decisions: list[bool | None] = []
p3_geometry_failures: list[str] = []
if p3_report is None:
    print("Итоговый отчёт p3 ещё не сохранён.")
else:
    print("Контроль p3:", p3_report["status"], "; завершённых входов:", p3_report["completed_input_runs"], "/16 (8 новых и 8 опорных).")
    for value in sequence(p3_report["cases"]):
        case = mapping(value)
        print(f"ECVB {case['model_id']} {case['kind']}: {case['status']}")
        for comparison_name in ("p3_vs_p4", "p3_vs_p5", "p3_dt"):
            comparison_value = case[comparison_name]
            if comparison_value is None:
                print("  ", comparison_name, "— ожидается")
                continue
            for window_value in sequence(mapping(comparison_value)["windows"]):
                window = mapping(window_value)
                if window["excluded_radius_m"] in (0.0, 20.0):
                    print("  ", comparison_name, window["time_window_s"], f"радиус {window['excluded_radius_m']} м:", component_percentages(mapping(window["components"])))
        for reference, criteria_value in mapping(case["criteria_by_reference"]).items():
            criteria = mapping(criteria_value)
            print(f"  Критерии относительно {reference}: {criteria['status']}")
            for name, component_value in mapping(criteria.get("components", {})).items():
                component = mapping(component_value)
                print(f"    {name}: абсолютное dt/p={component.get('dt_to_p_absolute_ratio')}; принято={component.get('accepted')}")
        for reference, force_value in mapping(case["source_force_by_reference"]).items():
            if force_value is not None:
                print(f"  Истории силы p3/{reference} одинаковы:", mapping(force_value)["identical_history"])
        for reference, localization_value in mapping(case["source_localization_by_reference"]).items():
            if localization_value is None:
                continue
            for name, component_value in mapping(mapping(localization_value)["components"]).items():
                component = mapping(component_value)
                print(f"  p3/{reference}, {name}: доля источника в квадрате разности={percent(component['source_share_of_squared_difference'])}; относительная разность в источнике={percent(component['source_only_relative_l2'])}; без единственного источникового приёмника={percent(component['excluding_only_source_relative_l2'])}")
    for value in sequence(p3_report["geometry_attribution_by_reference"]):
        attribution = mapping(value)
        for window_value in sequence(attribution.get("windows", [])):
            window = mapping(window_value)
            ratios = [number(mapping(item)["effect_over_correction_sum"]) for item in mapping(window["components"]).values() if mapping(item)["effect_over_correction_sum"] is not None]
            print(f"ECVB {attribution['model_id']}, p3 против p{attribution['reference_order']}, окно {window['time_window_s']}: минимум геометрическая разность / сумма p/dt-поправок =", min(ratios) if ratios else None)
            for name, component_value in mapping(window["components"]).items():
                component = mapping(component_value)
                margin_value = component["five_times_margin"]
                p3_geometry_decisions.append(margin_value if isinstance(margin_value, bool) else None)
                if margin_value is False:
                    p3_geometry_failures.append(f"ECVB {attribution['model_id']}, p3/p{attribution['reference_order']}, {name}, окно {window['time_window_s']}: отношение {number(component['effect_over_correction_sum']):.5g} <5")


"""
Сейсмограммы нового сравнения используют все четыре компоненты U/V. Дополнительные ролики сопоставляют raster/curved при p3 и p4 на общей постоянной шкале; исходные ролики p4/p5 остаются отдельным результатом. Дневник читает только подготовленные рисунки и описания анимаций.
"""

p3_comparison_path = p3_root / "comparison_all_v1/report.json"
p3_comparison_complete = False
if p3_comparison_path.is_file():
    p3_comparison = read_json(p3_comparison_path)
    p3_comparison_complete = p3_comparison.get("status") == "postprocessing_complete" and p3_comparison.get("completed_run_count") == 16
    for model_id, substeps in ((12404, 64), (65, 224)):
        filename = f"p_m{model_id:05d}_raster_p3_x350_k{substeps}_minus_m{model_id:05d}_raster_p5_x350_k{substeps}_full_far20.png"
        for value in sequence(p3_comparison.get("images", [])):
            if isinstance(value, str) and Path(value).name == filename:
                saved_image = PROJECT_ROOT / value
                if saved_image.is_file():
                    display(Image(filename=str(saved_image), width=1000))
    late_filename = "p_m00065_curved_p3_x350_k224_minus_m00065_curved_p5_x350_k224_late.png"
    for value in sequence(p3_comparison.get("images", [])):
        if isinstance(value, str) and Path(value).name == late_filename:
            saved_image = PROJECT_ROOT / value
            if saved_image.is_file():
                print("ECVB 65 curved, p3/p5: позднее окно 1…2 с, все приёмники; шкала общая внутри пары, у разности отдельная шкала.")
                display(Image(filename=str(saved_image), width=1000))
p3_animation_count = 0
for model_id in (12404, 65):
    animation_path = p3_root / f"animation_{model_id:05d}_p3_p4_v1/animation.json"
    if animation_path.is_file():
        animation = read_json(animation_path)
        if animation.get("status") == "animation_complete":
            p3_animation_count += 1
            print(f"ECVB {model_id}: анимация p3/p4, {animation['frame_count']} кадров.")
            for value in sequence(animation["outputs"]):
                if isinstance(value, str) and value.endswith(".mp4"):
                    print("Ролик:", value)
            preview = image_root / "central_p3_cu_v1" / f"animation_{model_id:05d}_p3_p4_v1/speed_t0_30.png"
            if not preview.is_file():
                preview = preview.with_name("speed_final.png")
            if preview.is_file():
                display(Image(filename=str(preview), width=1200))


""" ## 13. История и границы текущей программы ## """

"""
Прерванный `series_v1` с внешним SIGTERM и кодом 143 не является завершённым решением и исключён из сравнений. Старые планы `central_plan_v1` и `central_full_v1` не заменяют актуальную серию `central_cu_v1`; отдельный завершённый CEED-опыт используется только в проверке backend. Предыдущие рисунки и предварительные планы не перебираются автоматически и не дублируют итоговые артефакты.

Пять положений источника остаются предусмотрены геометрией, но по уточнению пользователя сейчас исследуется только x=350 м. Программа остальных четырёх положений отложена. Диагностика разрежения истории работы сохранена отдельно и не применялась: основная серия интегрирует работу по каждому внутреннему шагу. Контроль отражения от слоя, накопленные потери ABC/αM и замкнутый баланс основной области/слоя не выполнены.
"""

""" ## Выводы ## """

"""
Четыре расширенные Q2-сетки, включая новую curved №65, построены и геометрически проверены. Реализованы точечная сила 30 Гц, слой 175 м, запуск с фиксацией native-профиля, поверхностные U/V и всеузловые записи скорости/энергии. Результаты исполнения и научная интерпретация разделены: завершённый расчёт не означает автоматически прохождения рабочих критериев.
"""

all_runs_complete = completed_main == 8 and completed_controls == 4 and failed == 0
comparison_complete = final_comparison_ready and comparison_report is not None and comparison_report.get("status") == "postprocessing_complete" and comparison_report.get("completed_run_count") == 12
acceptance_complete = acceptance_report is not None and acceptance_report.get("completed_input_runs") == 12
all_reports_complete = comparison_complete and acceptance_complete and source_diagnostic_count == 4 and animation_count == 2
print(f"Фактическое исполнение исходной серии p4/p5: {completed_main}/8 основных опытов и {completed_controls}/4 временных контролей.")
if not all_runs_complete or not all_reports_complete:
    print("Глава пока содержит промежуточные результаты: полный расчётный пакет и/или его выбранная итоговая постобработка ещё не завершены.")
else:
    print("Все двенадцать расчётов и выбранные итоговые отчёты исходной центральной серии p4/p5 сохранены.")
if all_receiver_over_one_percent:
    print("p-разность по полной линии превышает 1% в следующих случаях:")
    for item in all_receiver_over_one_percent:
        print("  ", item)
    print("Поэтому выполнение критериев far20 не означает сходимости всей записи, включающей источник.")
if far20_pass is True:
    print("Для far20 на полном интервале 0…2 с все четыре случая выполняют рабочие критерии p≤1%, dt≤0,1% и абсолютного запаса 1:5.")
elif far20_pass is False:
    print("Рабочие критерии far20 выполнены не во всех случаях; безусловное принятие p4 по этой серии не подтверждено.")
else:
    print("Итог рабочих критериев far20 остаётся неопределённым: отсутствуют входы, отчёт или определимая опорная норма.")
print("Выводы относятся только к центральному источнику. Эффективность слоя, теоретическая сходимость источникового узла и независимая валидация решателя здесь не установлены.")
if p3_complete and p3_comparison_complete and p3_animation_count == 2:
    print("Дополнительная серия p3 завершена: четыре основных опыта, четыре временных контроля, сравнения с обеими опорами и анимации.")
    print("Рабочие критерии p3 относительно p4 и p5:", p3_report["status"] if p3_report is not None else "не определены")
    if len(p3_geometry_decisions) != 32 or any(value is None for value in p3_geometry_decisions):
        print("Отдельный пятикратный запас геометрической разности p3 ещё не определён полностью.")
    elif p3_geometry_failures:
        print("Отдельный пятикратный запас геометрической разности p3 выполнен не везде:")
        for item in p3_geometry_failures:
            print("  ", item)
        print("Прохождение критериев far20 само по себе не обосновывает замену p4 на p3 для исследования сглаживания.")
        print("Рекомендация для основной программы исследования сглаживания: сохранить p4.")
    else:
        print("Отдельный пятикратный запас геометрической разности p3 выполнен по обеим опорам в обоих окнах.")
else:
    print("Дополнительная проверка p3 ещё не завершена полностью; она не изменяет сохранённые итоги p4/p5.")
