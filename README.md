# czii-model-comparison

# Запуск экспериментов на виртуальной машине

## Победитель соревнования - Segmentation
1. Загрузить данные соревнования на ВМ.
1. Запустить [vm_segmentation.sh](competition_winner/vm_segmentation.sh) для подготовки окружения.
1. Перезапустить терминал и в нем запустить docker-контейнер командой:
    ```bash
    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.08-py3
    ```
1. Внутри контейнера запустить:
    ```bash
    cd kaggle-cryo-1st-place-segmentation
    pip install -r requirements.txt
    ```
1. Запустить тренировку:
    ```
    python train_segmentation.py -C cfg_resnet34 --fold -1
    python train_segmentation.py -C cfg_effnetb3 --fold -1
    ```

## Победитель соревнования - Object Detection
1. Загрузить данные соревнования на ВМ.
1. Запустить [vm_object_detection.sh](competition_winner/vm_object_detection.sh) для подготовки окружения.
1. Запустить тренировку:
```bash
cd Kaggle-2024-CryoET && bash train.sh
```

## DeePiCt
1. Загрузить данные соревнования на ВМ.
1. Запустить скрипт [vm_deepict.sh](deepict/vm_deepict.sh) для подготовки окружения.
1. Запустить скрипт [convert_data.py](deepict/convert_data.py) для конвертации данных под формат `DeePiCt`.
1. Запустить скрипт [create_masks.py](deepict/convert_data.py) для создания масок.
1. Запустить скрипт [generate_metadata.py](deepict/convert_data.py) для генерации таблицы с данными.
1. Запустить скрипт [deploy_local.sh](deepict/deploy_local.sh) для конфига каждого класса.
```bash
bash deploy_local.sh czii_config_<class_name>.yaml
```
