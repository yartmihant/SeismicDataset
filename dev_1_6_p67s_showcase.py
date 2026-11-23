""" # Глава 1.6. Семь источников на оптимизированном p67s # """

"""
Повторная визуальная серия для однородной, двуслойной и четырёх моделей EFVB/ECVB. Дневник читает только сохранённые результаты; не подключается к Saturn и не запускает решатель. Все 42 опыта независимы: семь состояний на каждой из шести геометрий.
"""

from pathlib import Path
from IPython.display import Image, display
from scripts.reporting import read_json, mapping
from scripts.project_paths import PROJECT_ROOT

chapter16_data = PROJECT_ROOT / "data/dev_1_6_p67s_showcase/series_v1"
chapter16_images = PROJECT_ROOT / "img/dev_1_6_p67s_showcase/series_v1"
chapter16_cases = ("homogeneous", "two_layer", "EFVB00000", "EFVB00001", "ECVB12404", "ECVB00065")

""" ## 1. Постановка ## """

"""
Основная область1400×700 м, полная со слоем2100×1050 м. Исходные EFVB/ECVB растянуты вдвое по горизонтали. ECVB12404 и00065 имеют сглаженные границы; EFVB00000 и00001 — исходные горизонтальные слои. Однородная среда: Vp4087, Vs2431.62036 м/с; в двухслойной нижний материал Vp3963, Vs2357.84473 м/с, интерфейс y=−350 м. Плотность1000 кг/м³, плоская деформация.

p6, целевой размер10 м, Q2-геометрия и49 GLL-точек на элемент. Весь верх —210 элементов по10 м и1261 узел. PML350 м по бокам/дну, направленный квадратичный профиль с максимумом173.0886108476381 с⁻¹. ABC и дополнительного массового демпфирования нет, граничное пространство не создаётся. Внешние края естественные свободные. Материал плоскопараллельно продолжается в слой.

Семь источников x175/350/525/700/875/1050/1225 м, y0: точечная сила Рикера вниз1 Н/м,30 Гц,t0=0.05 с. Сила одинакова, введённая энергия не калибруется. В каждом из семи независимых состояний действует только один источник.

Newmark–Kim β=1/27,γ=0.5,FP64,1 с. Native dt с cfl_safety0.8 отдельно для каждой геометрии, tact_dt0.001 с. Сохраняются21 кадр полного поля Vx/Vy через0.05 с и1001 отсчёт всех1261 узлов верха через0.001 с, включая PML. Полная постановка: docs/dev_1_6_p67s_showcase_spec.txt.
"""

chapter16_spec = (PROJECT_ROOT / "docs/dev_1_6_p67s_showcase_spec.txt").read_text()
print(chapter16_spec)

""" ## 2. Геометрии и фактические шаги ## """

for chapter16_case in chapter16_cases:
    chapter16_result_path = chapter16_data / chapter16_case / "run/result.json"
    if chapter16_result_path.exists():
        chapter16_result = read_json(chapter16_result_path)
        print(chapter16_case, {key: chapter16_result[key] for key in ("nodes", "elements", "dt_s", "steps", "run_seconds", "terminal_state_finite", "surface_equals_fields")})
    chapter16_material = chapter16_images / chapter16_case / "material.png"
    if chapter16_material.exists():
        display(Image(filename=str(chapter16_material), width=1100))

""" ## 3. Поля и сейсмограммы ## """

"""
Линейная шкала±1e−10 м/с общая для обеих компонент, всех моделей, источников и времён. Значения за пределом насыщают цвет, исходные массивы не меняются. Дополнительная знаковая логарифмическая шкала имеет единый максимум серии и линейный порог1e−13 м/с. Анимации5fps замедлены в4 раза. Поле для отображения линейно интерполируется по треугольникам GLL-сетки; сейсмограммы показывают исходные узловые отсчёты.

Ниже центральный источник для каждой модели; все семь источников и обе шкалы представлены в img/dev_1_6_p67s_showcase/series_v1/index.html и вложенных галереях. Галерея содержит отдельные MP4, PNG сейсмограмм и кадры t0.25/0.5 с для42 опытов.
"""

for chapter16_case in chapter16_cases:
    for chapter16_name in ("frame_0.25.png", "seismogram.png"):
        chapter16_picture = chapter16_images / chapter16_case / "linear/x0700" / chapter16_name
        if chapter16_picture.exists():
            display(Image(filename=str(chapter16_picture), width=1100))

""" ## 4. Проверки и ограничения ## """

"""
Проверяются положительность Якобиана на полной Q2-карте, конформность/топология, точные позиции источников, конечность кадров и терминального состояния/PML, побитовое совпадение поверхности и полей на21 общем времени. Native CFL не доказывает устойчивость полного спектра PML. Эта серия длительностью1 с не является независимой проверкой сеточной сходимости всех60 000 моделей. Тайминги включают два приёмника и ожидание их очередей; один запуск на геометрию, не медианный бенчмарк.
"""

chapter16_completion = chapter16_data / "completion.json"
if chapter16_completion.exists():
    print(read_json(chapter16_completion))
else:
    print("Полная серия ещё не подтверждена итоговым completion.json.")
