# Среда SeismicDataset

Проект использует Python 3.13 и собственную `.venv`. Пакет `semgpu2` подключается editable по пути `../../SEM/TrySEMGPU/semgpu2`; первый пакет `semgpu` этому исследованию не нужен. Местоположение `src/` относится только к исходным данным, не к Python-пакетам.

## Отчёты и подготовка данных

```bash
SEMGPU2_BUILD_CUDA=0 SEMGPU2_BUILD_CEED=0 uv sync --locked --extra reports --extra dev --inexact --config-settings-package "semgpu2:build-dir=$PWD/.venv/build/semgpu2-base/{wheel_tag}"
.venv/bin/python -m mypy scripts dev_1_1_dataset_preprocessing.py dev_1_2_smoothing_pilot.py
MPLBACKEND=Agg .venv/bin/python dev_1_1_dataset_preprocessing.py
.venv/bin/python dev_1_2_smoothing_pilot.py
```

Команда установки задаёт отдельный каталог базовой сборки; `--inexact` сохраняет отдельно установленный инструмент авторинга py2jupyter.

На Windows использовать `.venv\Scripts\python.exe`. Базовый `semgpu2` подходит для чтения модели, геометрии и отчётного анализа; он не содержит CU/CEED native-модулей. Не считать успешный отчётный запуск проверкой GPU.

## Native CU и CEED

Сборка выполняется по [контракту SEMGPU](../../../SEM/TrySEMGPU/docs/BUILD_RUN_ENVIRONMENT.md). Параметры CUDA-архитектуры, libCEED и компиляторов выбираются для конкретной машины. Использовать отдельный build-dir этого проекта, чтобы не разделять изменяемый native build с основной средой TrySEMGPU.

Пример только для CUDA-профиля Linux с архитектурой 86:

```bash
SEMGPU2_BUILD_CUDA=1 SEMGPU2_BUILD_CEED=0 uv pip install --python .venv/bin/python --reinstall -e ../../SEM/TrySEMGPU/semgpu2 -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86 -Cbuild-dir="$PWD/.venv/build/semgpu2-cuda/{wheel_tag}"
```

Для CEED требуется отдельный подготовленный prefix и `SEMGPU2_BUILD_CEED=1`; Windows-утилиты принимают явный путь DLL/JIT-компилятора. Доступность native проверять свежим процессом и фиксировать origin/SHA256. Обновление Python-исходников через editable не пересобирает native автоматически. После обычного `uv sync` проверять установленный native-профиль, не предполагать его сохранение по одному наличию пакета.

## Конвертация дневников

Используется тот же пользовательский `py2jupyter`, что в MilenSem2D. При наличии локального проекта инструмента:

```bash
uv pip install --python .venv/bin/python -e /home/antonov/Work/Tools/Org/PyToJupyter
.venv/bin/python -m ipykernel install --prefix .venv --name seismic-dataset --display-name "SeismicDataset (.venv)"
.venv/bin/python -m scripts.build_notebooks --execute
```

Обёртка `scripts/build_notebooks.py` использует исправленный py2jupyter, проверяет синтаксис, nbformat и сохранение AST при обратной конвертации. Исправление сохранения LF внесено непосредственно в PyToJupyter по отдельному поручению пользователя; локальная подмена source-строк больше не нужна. Ключ `--execute` выполняет три отчётных дневника; `--diary dev_1_3_point_source` выбирает только главу 1.3. Без `--execute` создаётся представление без outputs.

## Профиль главы 1.3

13 ноября 2026 года отдельно собран CEED-native в `.venv/build/semgpu2-ceed/cp313-cp313-linux_x86_64`, Release, scikit-build-core1.0.3, pybind11 3.1.0, два задания сборки. Использован существующий libCEED-prefix `/home/antonov/Work/Areas/SEM/TrySEMGPU/.venv/libceed/p113-cuda`; исходники SEMGPU не изменялись. Профиль, Git revision чистого checkout, версии и SHA256 binding/libCEED записаны в `data/dev_1_3_point_source/runtime_v1/profile.json`.

Короткие проверки `/cpu/self/opt/blocked` и `/gpu/cuda/gen` выполнены на p4/p5; GPU — RTX3060 12GiB, драйвер580.196.07. В ограниченной среде обычных команд устройства GPU скрыты; отсутствие `/dev/nvidia*` в ней не означает отсутствия видеокарты на машине. Для GPU-прогонов нужен доступ к устройству. Ни совпадение CPU/GPU-smoke, ни успешная сборка не заменяют сходимость новой серии.

Затем штатно установлен combined CUDA+CEED пакет в `.venv/build/semgpu2-combined`, nvcc12.8, arch86, два задания сборки. Старые CEED binding/libCEED сохранили SHA256; профиль и проверка установки — `data/dev_1_3_point_source/runtime_combined_v1/`. CU-проверка полной сетки на 0,01 с согласовалась с CEED до округления и была быстрее; основной центральный план `central_cu_v1` целиком использует CU. Полный CEED-опыт остаётся отдельным сравнительным результатом. Не смешивать backend между p4/p5 в одной основной паре.

Длинные расчёты запускаются отдельно от времени жизни одной команды. Состояние очереди и отдельных запусков фиксируется в `process.log`, `batch_events/` и `metadata*.json`; отсутствие успешного `metadata.json` означает отсутствие подтверждённого завершения. Повтор использует новую серию либо пропускает только полностью проверенные завершённые результаты. Частичные каталоги не удаляются автоматически.

На другой машине путь к инструменту задаётся по фактическому checkout. Это инструмент авторинга, не runtime-зависимость расчёта или чтения готового ноутбука. Обратную конвертацию для проверки делать во временный файл, не поверх первичного дневника.

## Воспроизводимость

`uv.lock` фиксирует внешние Python-зависимости. Для расчётной серии дополнительно сохранять Git revision и diff пакета SEMGPU, параметры сборки, native SHA256, сведения о Python, GPU/драйвере и libCEED. Изменяемый соседний checkout сам по себе не является замороженной зависимостью. Исторический native-снимок в `data/` предназначен только для проверки происхождения уже полученных результатов.

Для новых бинарных серий `SEISMIC_DATASET_BINARY` задаёт каталог блоков и manifest вместо исторического `data/dev_1_1_preprocessing/binary`. Например: `export SEISMIC_DATASET_BINARY="$PWD/data/dev_1_1_preprocessing/binary_new"`. Эта настройка применяется к подготовке геометрии и статистике; отчётные дневники продолжают читать явно указанные исторические результаты.
