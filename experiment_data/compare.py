import os
import glob
import pandas as pd
import numpy as np
import re
import json
import random # Добавлено для случайного выбора срезов
import zarr # Добавлено для чтения томограмм в формате Zarr
from scipy.spatial import KDTree
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D # Больше не нужно для 3D графиков
import seaborn as sns
from tqdm import tqdm  # Добавлено для отображения прогресса

def softmax(x):
    """
    Применяет функцию softmax для преобразования логитов в вероятности.
    
    Параметры:
    -----------
    x : numpy.ndarray или list
        Массив логитов
        
    Возвращает:
    --------
    numpy.ndarray
        Массив вероятностей
    """
    x = np.array(x, dtype=np.float32)  # Убедиться, что входные данные - numpy массив
    # Для логитов 1.0 и 0.0 мы хотим получить вероятности, близкие к 1.0 и 0.0
    # Поэтому масштабируем логиты, чтобы разница была более выраженной
    x = x * 5  # Увеличиваем логиты, чтобы сделать разницу более значимой
    exp_x = np.exp(x - np.max(x))  # Вычитаем максимум для численной стабильности
    return exp_x / exp_x.sum()

def load_deepict_predictions(base_dir="csv_outputs/deepict"):
    """
    Преобразует предсказания из модели deepict в pandas dataframe с колонками:
    [tomo,x,y,z,prob_beta_amylase,prob_beta_galactosidase,prob_ribosome,
    prob_thyroglobulin,prob_virus_like_particle,prob_apo_ferritin]
    
    Параметры:
    -----------
    base_dir : str
        Путь к директории, содержащей выходные CSV файлы deepict
        
    Возвращает:
    --------
    pandas.DataFrame
        DataFrame с распределениями вероятностей для всех классов
    """
    # Получить все CSV файлы из директории deepict
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    
    # Отфильтровать файл с кривой обучения
    csv_files = [f for f in csv_files if "learning_curve" not in f]
    
    if not csv_files:
        raise FileNotFoundError(f"Файлы предсказаний CSV не найдены в {base_dir}")
    
    # Инициализация словаря для хранения координат и логитов
    all_coords = {}
    logits_by_class = {}
    
    # Определение имен классов в порядке
    class_names = ['beta_amylase', 'beta_galactosidase', 'ribosome', 
                  'thyroglobulin', 'virus_like_particle', 'apo_ferritin']
    
    # Парсинг каждого CSV файла
    for csv_file in csv_files:
        # Извлечение имени класса из имени файла
        filename = os.path.basename(csv_file)
        match = re.search(r'TS_\d+_\d+_([a-z-]+)\.csv', filename)
        
        if not match:
            continue
            
        class_name = match.group(1).replace('-', '_')
        
        # Чтение CSV файла (без заголовка, формат x,y,z,value)
        df = pd.read_csv(csv_file, header=None)
        df.columns = ['x', 'y', 'z', 'logit']
        
        # Извлечение имени томограммы из имени файла (например, TS_99_9)
        tomo_match = re.search(r'(TS_\d+_\d+)', filename)
        tomo_name = tomo_match.group(1) if tomo_match else "unknown"
        
        # Для каждой координаты сохраняем значение логита
        coords_key = lambda row: (tomo_name, row['x'], row['y'], row['z'])
        
        for _, row in df.iterrows():
            key = coords_key(row)
            if key not in all_coords:
                all_coords[key] = True
                # Инициализируем логиты для всех классов как -inf (или очень отрицательное число)
                logits_by_class[key] = {name: -100.0 for name in class_names}
            
            # Сохраняем значение логита для этого класса
            logits_by_class[key][class_name] = row['logit']
    
    # Создаем финальный dataframe
    result_data = []
    
    for key in all_coords.keys():
        tomo, x, y, z = key
        
        # Получаем логиты для всех классов в этой координате
        logits = [logits_by_class[key][name] for name in class_names]
        
        # Преобразуем логиты в вероятности с помощью softmax
        probs = softmax(logits)
        
        # Создаем данные строки
        row_data = {
            'tomo': tomo,
            'x': x,
            'y': y,
            'z': z
        }
        
        # Добавляем вероятности в данные строки
        for class_name, prob in zip(class_names, probs):
            prob_key = f'prob_{class_name}'
            row_data[prob_key] = prob
        
        result_data.append(row_data)
    
    # Создаем финальный dataframe
    result_df = pd.DataFrame(result_data)
    
    # Убеждаемся, что колонки в правильном порядке
    column_order = ['tomo', 'x', 'y', 'z', 
                   'prob_beta_amylase', 'prob_beta_galactosidase', 
                   'prob_ribosome', 'prob_thyroglobulin', 
                   'prob_virus_like_particle', 'prob_apo_ferritin']
    
    return result_df[column_order]

def competition_predictions_to_dataframe(model_type, model_name):
    """
    Преобразует предсказания из других моделей (object_detection и segmentation) 
    в pandas dataframe с колонками:
    [tomo,x,y,z,prob_beta_amylase,prob_beta_galactosidase,prob_ribosome,
    prob_thyroglobulin,prob_virus_like_particle,prob_apo_ferritin]
    
    Параметры:
    -----------
    model_type : str
        Тип модели: 'object_detection' или 'segmentation'
    model_name : str
        Название модели: 'dynunet', 'segresnetv2', 'resnet34', или 'effnetb3'
        
    Возвращает:
    --------
    pandas.DataFrame
        DataFrame с распределениями вероятностей для всех классов
    """
    base_dir = f"csv_outputs/{model_type}/{model_name}"
    
    # Найти выходной CSV файл (не постобработанный)
    csv_files = glob.glob(os.path.join(base_dir, "*preds.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Файлы предсказаний CSV не найдены в {base_dir}")
    
    # Использовать первый найденный файл
    csv_file = csv_files[0]
    
    # Прочитать CSV файл
    df = pd.read_csv(csv_file)
    
    # Извлечь координаты и добавить колонку tomo (предполагается одинаковой для всех предсказаний)
    df['tomo'] = 'TS_99_9'
    
    # Получить все колонки с логитами
    logit_cols = [col for col in df.columns if col.startswith('logits_')]
    
    # Преобразовать логиты в вероятности с использованием softmax
    for idx, row in df.iterrows():
        # Получить логиты как numpy массив
        logits = np.array([row[col] for col in logit_cols], dtype=np.float32)
        probs = softmax(logits)
        
        # Обновить строку вероятностями
        for i, col in enumerate(logit_cols):
            prob_col = col.replace('logits_', 'prob_')
            df.at[idx, prob_col] = probs[i]
    
    # Убедиться, что колонки в правильном порядке
    column_order = ['tomo', 'x', 'y', 'z', 
                   'prob_beta_amylase', 'prob_beta_galactosidase', 
                   'prob_ribosome', 'prob_thyroglobulin', 
                   'prob_virus_like_particle', 'prob_apo_ferritin']
    
    # Выбрать только требуемые колонки
    result_df = df[column_order]
    
    return result_df

def load_competition_predictions():
    """
    Получить предсказания от всех моделей и объединить их в отдельные dataframes.
    
    Возвращает:
    --------
    dict
        Словарь с именами моделей в качестве ключей и dataframes в качестве значений
    """
    models = [
        competition_predictions_to_dataframe('object_detection', 'dynunet'),
        competition_predictions_to_dataframe('object_detection', 'segresnetv2'),
        competition_predictions_to_dataframe('segmentation', 'resnet34'),
        competition_predictions_to_dataframe('segmentation', 'effnetb3')
    ]
    
    return models

def load_ground_truths(base_dir="ground_truths"):
    """
    Загрузить данные истинных меток из JSON файлов и преобразовать в pandas dataframe с колонками:
    [tomo,x,y,z,prob_beta_amylase,prob_beta_galactosidase,prob_ribosome,
    prob_thyroglobulin,prob_virus_like_particle,prob_apo_ferritin]
    
    Параметры:
    -----------
    base_dir : str
        Путь к директории с JSON-файлами истинных меток
        
    Возвращает:
    --------
    pandas.DataFrame
        DataFrame с точками истинных меток и их вероятностями классов
    """
    # Получить все JSON файлы из директории истинных меток
    json_files = glob.glob(os.path.join(base_dir, "*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"JSON-файлы истинных меток не найдены в {base_dir}")
    
    # Инициализировать словарь для хранения всех точек
    all_points = {}
    
    # Обработать каждый JSON файл
    for json_file in json_files:
        # Извлечь имя класса из имени файла
        class_name = os.path.basename(json_file).replace('.json', '').replace('-', '_')
        
        # Прочитать JSON файл
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Получить имя томограммы из run_name
        tomo_name = data.get('run_name', 'TS_99_9')
        
        # Обработать каждую точку
        for point in data.get('points', []):
            location = point.get('location', {})
            x, y, z = location.get('x', 0), location.get('y', 0), location.get('z', 0)
            
            # Создать уникальный ключ для этой точки
            point_key = (tomo_name, x, y, z)
            
            if point_key not in all_points:
                all_points[point_key] = {
                    'tomo': tomo_name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'prob_beta_amylase': 0.0,
                    'prob_beta_galactosidase': 0.0,
                    'prob_ribosome': 0.0,
                    'prob_thyroglobulin': 0.0,
                    'prob_virus_like_particle': 0.0,
                    'prob_apo_ferritin': 0.0
                }
            
            # Установить вероятность 1.0 для текущего класса
            prob_key = f'prob_{class_name}'
            all_points[point_key][prob_key] = 1.0
    
    # Преобразовать словарь в DataFrame
    result_df = pd.DataFrame(list(all_points.values()))
    
    # Убедиться, что колонки в правильном порядке
    column_order = ['tomo', 'x', 'y', 'z', 
                   'prob_beta_amylase', 'prob_beta_galactosidase', 
                   'prob_ribosome', 'prob_thyroglobulin', 
                   'prob_virus_like_particle', 'prob_apo_ferritin']
    
    return result_df[column_order]

def compute_metrics(reference_points, reference_radius, candidate_points):
    """
    Вычисляет истинно-положительные, ложно-положительные и ложно-отрицательные результаты между опорными и кандидатными точками.
    
    Параметры:
    -----------
    reference_points : numpy.ndarray
        Массив координат опорных точек
    reference_radius : float
        Максимальное расстояние для рассмотрения точек как совпадений
    candidate_points : numpy.ndarray
        Массив координат точек-кандидатов
        
    Возвращает:
    --------
    tuple
        (истинно_положительные, ложно_положительные, ложно_отрицательные)
    """
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Предотвращает отправку нескольких совпадений для одной частицы
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn

def compute_fbeta_score(prediction_df, ground_truth_df, beta=1, distance_multiplier=1.0, thresholds=None):
    """
    Вычисляет оценку F-beta между dataframe предсказаний и dataframe истинных меток.
    
    Параметры:
    -----------
    prediction_df : pandas.DataFrame
        DataFrame с вероятностями предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame с вероятностями истинных меток
    beta : float
        Параметр бета для оценки F-beta (по умолчанию: 1 для оценки F1)
    distance_multiplier : float
        Множитель для пороговых значений радиуса частиц
    thresholds : dict, optional
        Словарь с пороговыми значениями для каждого класса (по умолчанию: 0.5 для всех классов)
        
    Возвращает:
    --------
    dict
        Словарь, содержащий:
        - 'weighted_score': Общая взвешенная оценка F-beta
        - 'class_scores': Словарь индивидуальных оценок классов
        - 'metrics': Словарь, содержащий точность, полноту и F-beta для каждого класса
    """
    # Определить радиусы частиц и веса
    particle_radius = {
        'apo_ferritin': 60,
        'beta_amylase': 65,
        'beta_galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus_like_particle': 135,
    }

    weights = {
        'apo_ferritin': 1,
        'beta_amylase': 0,
        'beta_galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus_like_particle': 1,
    }

    # Использовать пороговое значение по умолчанию (0.5), если не предоставлено
    if thresholds is None:
        thresholds = {k: 0.5 for k in particle_radius.keys()}

    # Применить множитель расстояния к радиусам
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Получить уникальные томограммы
    tomos = set(ground_truth_df['tomo'].unique())
    prediction_df = prediction_df[prediction_df['tomo'].isin(tomos)]

    # Инициализировать словарь результатов
    results = {}
    for particle_type in particle_radius.keys():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    # Обработать каждую томограмму и тип частиц
    for tomo in tomos:
        for particle_type in particle_radius.keys():
            # Получить опорные точки (истинные метки)
            gt_mask = (ground_truth_df['tomo'] == tomo) & (ground_truth_df[f'prob_{particle_type}'] > 0.5)
            reference_points = ground_truth_df.loc[gt_mask, ['x', 'y', 'z']].values

            # Получить точки-кандидаты (предсказания), используя порог для этого класса
            threshold = thresholds.get(particle_type, 0.5)
            pred_mask = (prediction_df['tomo'] == tomo) & (prediction_df[f'prob_{particle_type}'] > threshold)
            candidate_points = prediction_df.loc[pred_mask, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1
            else:
                reference_radius = particle_radius[particle_type]

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            # Вычислить метрики
            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    # Вычислить оценки для каждого класса
    class_scores = {}
    metrics = {}
    aggregate_fbeta = 0.0
    
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        
        class_scores[particle_type] = fbeta
        metrics[particle_type] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    # Нормализовать по сумме весов
    aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    
    return {
        'weighted_score': aggregate_fbeta,
        'class_scores': class_scores,
        'metrics': metrics
    }

def find_optimal_thresholds(prediction_df, ground_truth_df, beta=1, distance_multiplier=1.0):
    """
    Находит оптимальные пороговые значения для каждого класса, которые максимизируют оценки F-beta.
    
    Параметры:
    -----------
    prediction_df : pandas.DataFrame
        DataFrame с вероятностями предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame с вероятностями истинных меток
    beta : float
        Параметр бета для оценки F-beta
    distance_multiplier : float
        Множитель для пороговых значений радиуса частиц
        
    Возвращает:
    --------
    dict
        Словарь с оптимальными пороговыми значениями для каждого класса
    """
    # Определить типы частиц
    particle_types = [col.replace('prob_', '') for col in prediction_df.columns if col.startswith('prob_')]
    
    # Инициализировать оптимальные пороговые значения
    optimal_thresholds = {}
    
    # Попробовать различные пороговые значения для каждого класса
    threshold_values = np.linspace(0.1, 0.9, 9)
    
    for particle_type in particle_types:
        best_score = -1
        best_threshold = 0.5
        
        for threshold in threshold_values:
            # Создать словарь пороговых значений с 0.5 по умолчанию для всех классов,
            # кроме текущего оптимизируемого
            thresholds = {pt: 0.5 for pt in particle_types}
            thresholds[particle_type] = threshold
            
            # Вычислить оценку F-beta с этим порогом
            scores = compute_fbeta_score(
                prediction_df, ground_truth_df, beta=beta,
                distance_multiplier=distance_multiplier, thresholds=thresholds
            )
            
            # Проверить, дает ли этот порог лучшую оценку для этого класса
            if scores['class_scores'].get(particle_type, 0) > best_score:
                best_score = scores['class_scores'].get(particle_type, 0)
                best_threshold = threshold
        
        optimal_thresholds[particle_type] = best_threshold
    
    return optimal_thresholds

def compute_bootstrap_intervals(prediction_df, ground_truth_df, n_bootstraps=100, beta=1, 
                               thresholds=None, confidence_level=0.95, 
                               distance_multiplier=1.0, exclude_beta_amylase=True):
    """
    Вычисляет доверительные интервалы bootstrap для оценок F-beta путем повторной выборки точек.
    
    Параметры:
    -----------
    prediction_df : pandas.DataFrame
        DataFrame с вероятностями предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame с вероятностями истинных меток
    n_bootstraps : int
        Количество bootstrap-выборок для генерации
    beta : float
        Параметр бета для оценки F-beta
    thresholds : dict, optional
        Словарь с пороговыми значениями для каждого класса
    confidence_level : float
        Уровень доверия для интервалов (например, 0.95 для 95% доверия)
    distance_multiplier : float
        Множитель для пороговых значений радиуса частиц
    exclude_beta_amylase : bool
        Если True, исключить beta_amylase из результатов
        
    Возвращает:
    --------
    dict
        Словарь с нижними и верхними границами для каждого класса и общей оценки
    """
    # Определить типы частиц и радиусы
    particle_types = [col.replace('prob_', '') for col in prediction_df.columns if col.startswith('prob_')]
    particle_radius = {
        'apo_ferritin': 60,
        'beta_amylase': 65,
        'beta_galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus_like_particle': 135,
    }
    
    # Применить множитель расстояния к радиусам
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}
    
    # Определить веса для классов (так же, как и в compute_fbeta_score)
    weights = {
        'apo_ferritin': 1,
        'beta_amylase': 0,
        'beta_galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus_like_particle': 1,
    }
    
    if exclude_beta_amylase and 'beta_amylase' in particle_types:
        particle_types.remove('beta_amylase')
    
    # Инициализировать массивы для хранения результатов bootstrap
    bootstrap_scores = {
        'weighted_score': np.zeros(n_bootstraps),
    }
    
    for particle_type in particle_types:
        bootstrap_scores[particle_type] = np.zeros(n_bootstraps)
    
    # Настройка для расчета доверительного интервала
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    # Запустить итерации bootstrap
    for i in tqdm(range(n_bootstraps), desc="Bootstrap sampling"):
        # Инициализировать метрики для этой итерации bootstrap
        particle_metrics = {}
        
        # Обработать каждый тип частиц отдельно
        for particle_type in particle_types:
            # Пропустить beta_amylase, если исключено
            if particle_type == 'beta_amylase' and exclude_beta_amylase:
                continue
                
            # Получить порог для этого класса
            threshold = thresholds.get(particle_type, 0.5) if thresholds else 0.5
            
            # Получить точки истинных меток для этого класса
            gt_mask = ground_truth_df[f'prob_{particle_type}'] > 0.5
            gt_points = ground_truth_df.loc[gt_mask, ['x', 'y', 'z']].values
            
            # Продолжить только если у нас есть точки истинных меток
            if len(gt_points) == 0:
                particle_metrics[particle_type] = {'tp': 0, 'fp': 0, 'fn': 0}
                continue
                
            # Bootstrap выборка точек истинных меток с заменой
            bootstrap_indices = np.random.choice(len(gt_points), size=len(gt_points), replace=True)
            bootstrap_gt_points = gt_points[bootstrap_indices]
            
            # Получить точки предсказаний выше порога
            pred_mask = prediction_df[f'prob_{particle_type}'] > threshold
            pred_points = prediction_df.loc[pred_mask, ['x', 'y', 'z']].values
            
            # Пропустить, если нет предсказаний
            if len(pred_points) == 0:
                particle_metrics[particle_type] = {'tp': 0, 'fp': len(bootstrap_gt_points), 'fn': 0}
                continue
                
            # Bootstrap выборка точек предсказаний с заменой
            bootstrap_indices = np.random.choice(len(pred_points), size=len(pred_points), replace=True)
            bootstrap_pred_points = pred_points[bootstrap_indices]
            
            # Вычислить метрики, используя bootstrap точки
            tp, fp, fn = compute_metrics(
                bootstrap_gt_points, 
                particle_radius[particle_type], 
                bootstrap_pred_points
            )
            
            # Сохранить метрики
            particle_metrics[particle_type] = {'tp': tp, 'fp': fp, 'fn': fn}
        
        # Вычислить оценки F-beta для каждого класса
        weighted_sum = 0
        total_weight = 0
        
        for particle_type in particle_types:
            if particle_type not in particle_metrics:
                continue
                
            metrics = particle_metrics[particle_type]
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            # Вычислить точность и полноту
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            
            # Вычислить F-beta
            fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if precision + recall > 0 else 0
            
            # Сохранить в результатах
            bootstrap_scores[particle_type][i] = fbeta
            
            # Обновить взвешенную оценку
            weight = weights.get(particle_type, 1)
            weighted_sum += fbeta * weight
            total_weight += weight
        
        # Вычислить взвешенную оценку
        bootstrap_scores['weighted_score'][i] = weighted_sum / total_weight if total_weight > 0 else 0
    
    # Вычислить доверительные интервалы
    confidence_intervals = {}
    for metric, values in bootstrap_scores.items():
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals

def format_model_comparison(model_predictions, ground_truth_df, beta=1, exclude_beta_amylase=True, use_optimal_thresholds=False):
    """
    Форматирует результаты сравнения моделей в удобном табличном формате.
    
    Параметры:
    -----------
    model_predictions : dict
        Словарь, сопоставляющий имена моделей с их dataframes предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame истинных меток
    beta : float
        Параметр бета для оценки F-beta
    exclude_beta_amylase : bool
        Если True, класс beta_amylase будет исключен из результатов
    use_optimal_thresholds : bool
        Если True, находит и использует оптимальные пороговые значения для каждой модели/класса
        
    Возвращает:
    --------
    tuple
        (formatted_df, thresholds_dict) где:
        - formatted_df: pandas DataFrame с оценками сравнения
        - thresholds_dict: словарь, сопоставляющий имена моделей с их оптимальными пороговыми значениями
    """
    import pandas as pd
    
    # Словарь для хранения оптимальных пороговых значений для каждой модели
    all_model_thresholds = {}
    
    # Вычислить оценки для каждой модели
    results = {}
    for model_name, pred_df in model_predictions.items():
        # Найти оптимальные пороговые значения, если запрошено
        if use_optimal_thresholds:
            thresholds = find_optimal_thresholds(pred_df, ground_truth_df, beta=beta)
            all_model_thresholds[model_name] = thresholds
            scores = compute_fbeta_score(pred_df, ground_truth_df, beta=beta, thresholds=thresholds)
        else:
            scores = compute_fbeta_score(pred_df, ground_truth_df, beta=beta)
        
        model_results = {
            'Общая оценка': scores['weighted_score'],
            **scores['class_scores']
        }
        
        # Удалить beta_amylase, если запрошено
        if exclude_beta_amylase and 'beta_amylase' in model_results:
            del model_results['beta_amylase']
            
        results[model_name] = model_results
    
    # Преобразовать в DataFrame
    df = pd.DataFrame(results).T
    
    # Округлить все оценки до 4 десятичных знаков
    df = df.round(4)
    
    # Сортировать по общей оценке
    df = df.sort_values('Общая оценка', ascending=False)
    
    return df, all_model_thresholds

def print_model_comparison(model_predictions, ground_truth_df, beta=1, exclude_beta_amylase=True, 
                           use_optimal_thresholds=True, compute_ci=False, n_bootstraps=100):
    """
    Генерирует результаты сравнения моделей и графики оценок F-beta.
    
    Параметры:
    -----------
    model_predictions : dict
        Словарь, сопоставляющий имена моделей с их dataframes предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame истинных меток
    beta : float
        Параметр бета для оценки F-beta (по умолчанию=1 для оценки F1)
    exclude_beta_amylase : bool
        Если True, класс beta_amylase будет исключен из таблицы результатов
    use_optimal_thresholds : bool
        Если True, находит и использует оптимальные пороговые значения для каждой модели/класса (по умолчанию=True)
    compute_ci : bool
        Если True, вычисляет доверительные интервалы bootstrap для оценок
    n_bootstraps : int
        Количество bootstrap-выборок для использования в доверительных интервалах
        
    Возвращает:
    --------
    tuple
        (comparison_table, thresholds_dict, fig, confidence_intervals) где:
        - comparison_table: pandas DataFrame с оценками сравнения моделей
        - thresholds_dict: словарь с оптимальными пороговыми значениями для каждой модели и класса
        - fig: matplotlib Figure с оценками F-beta по классам
        - confidence_intervals: словарь с доверительными интервалами (если compute_ci=True)
    """
    # Сгенерировать таблицы сравнения
    df_excluded, thresholds_excluded = format_model_comparison(
        model_predictions, ground_truth_df, beta, exclude_beta_amylase, use_optimal_thresholds
    )
    df_all, thresholds_all = format_model_comparison(
        model_predictions, ground_truth_df, beta, exclude_beta_amylase=False, use_optimal_thresholds=use_optimal_thresholds
    )
    
    # Выбрать таблицу и пороговые значения для построения и возврата
    df_to_plot = df_excluded if exclude_beta_amylase else df_all
    thresholds_to_return = thresholds_excluded if exclude_beta_amylase else thresholds_all
    
    # Вычислить доверительные интервалы, если запрошено
    confidence_intervals = {}
    if compute_ci:
        print("Вычисление доверительных интервалов bootstrap...")
        for model_name, pred_df in model_predictions.items():
            # Использовать оптимальные пороговые значения модели, если доступны
            thresholds = thresholds_to_return.get(model_name, None) if use_optimal_thresholds else None
            
            confidence_intervals[model_name] = compute_bootstrap_intervals(
                pred_df, ground_truth_df, n_bootstraps=n_bootstraps, beta=beta,
                thresholds=thresholds, exclude_beta_amylase=exclude_beta_amylase
            )
    
    # Создать DataFrame в длинном формате для seaborn, который лучше работает для статистических графиков
    long_data = []
    
    for model in df_to_plot.index:
        for metric in df_to_plot.columns:
            score = df_to_plot.loc[model, metric]
            
            # Добавить доверительный интервал, если доступен
            ci_low, ci_high = None, None
            if compute_ci and model in confidence_intervals and metric in confidence_intervals[model]:
                ci_low, ci_high = confidence_intervals[model][metric]
            
            long_data.append({
                'Model': model,
                'Metric': metric,
                'Score': score,
                'CI_Low': ci_low,
                'CI_High': ci_high
            })
    
    long_df = pd.DataFrame(long_data)
    
    # Создать график с использованием seaborn
    plt.figure(figsize=(12, 7))
    
    # Установить стиль seaborn
    sns.set_style("whitegrid")
    
    # Следуя руководству из seaborn.pydata.org/tutorial/error_bars.html
    if compute_ci:
        # Создать стандартный столбчатый график без встроенных планок погрешностей
        ax = sns.barplot(
            x='Metric', 
            y='Score', 
            hue='Model', 
            data=long_df
        )
        
        # Вручную добавить планки погрешностей к каждому столбцу
        # Сначала получить все столбцы из графика
        bars = ax.patches
        
        # Количество столбцов на позицию x
        n_models = len(df_to_plot)
        
        # Ширина каждого столбца (необходима для позиционирования планок погрешностей)
        bar_width = bars[0].get_width() if bars else 0.8 / n_models
        
        # Добавить планки погрешностей отдельно
        for i, row in long_df.iterrows():
            if row['CI_Low'] is not None and row['CI_High'] is not None:
                # Получить позицию x для этого столбца
                # Столбцы сгруппированы по Metric, затем по Model внутри каждой метрики
                model_idx = list(long_df['Model'].unique()).index(row['Model'])
                metric_idx = list(long_df['Metric'].unique()).index(row['Metric'])
                
                # Вычислить индекс столбца
                bar_idx = metric_idx * n_models + model_idx
                
                if bar_idx < len(bars):
                    bar = bars[bar_idx]
                    x_pos = bar.get_x() + bar.get_width() / 2
                    
                    # Убедиться, что границы CI находятся в пределах диапазона [0, 1] для F-оценок
                    ci_low = max(0, min(row['CI_Low'], row['Score']))
                    ci_high = min(1, max(row['CI_High'], row['Score']))
                    
                    # Вычислить длины планок погрешностей (убедиться, что они положительные)
                    lower_err = max(0, row['Score'] - ci_low)
                    upper_err = max(0, ci_high - row['Score'])
                    
                    # Добавить планки погрешностей
                    plt.errorbar(
                        x=x_pos,
                        y=row['Score'],
                        yerr=[[lower_err], [upper_err]],
                        fmt='none',
                        color='black',
                        capsize=5,
                        elinewidth=1.5
                    )
    else:
        # Простой столбчатый график без планок погрешностей
        ax = sns.barplot(
            x='Metric', 
            y='Score', 
            hue='Model', 
            data=long_df
        )
    
    # Настроить график
    plt.ylabel(f'Fβ с β={beta}')
    plt.xlabel('Класс')
    plt.ylim(0, 1.0)
    plt.title('Сравнение производительности моделей')
    plt.legend(title='Модель', loc='lower right')
    
    # Повернуть метки оси x для лучшей читаемости
    plt.xticks(rotation=15)
    
    # Настроить макет
    plt.tight_layout()
    
    # Вернуть фигуру
    fig = plt.gcf()
    
    # Вернуть таблицу, пороговые значения, график и доверительные интервалы
    if compute_ci:
        return df_to_plot, thresholds_to_return, fig, confidence_intervals
    else:
        return df_to_plot, thresholds_to_return, fig

def plot_roc_curves(prediction_df, ground_truth_df, distance_multiplier=1.0):
    """
    Строит кривые ROC для каждого класса, сравнивая вероятности предсказаний с истинными метками.
    Использует KDTree для сопоставления точек в пределах радиуса частицы, аналогично compute_fbeta_score.
    
    Параметры:
    -----------
    prediction_df : pandas.DataFrame
        DataFrame с вероятностями предсказаний
    ground_truth_df : pandas.DataFrame
        DataFrame с вероятностями истинных меток
    distance_multiplier : float
        Множитель для пороговых значений радиуса частиц
    """
    # Определить радиусы частиц
    particle_radius = {
        'apo_ferritin': 60,
        'beta_amylase': 65,
        'beta_galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus_like_particle': 135,
    }
    
    # Применить множитель расстояния к радиусам
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}
    
    # Получить имена классов из колонок вероятностей
    class_names = [col.replace('prob_', '') for col in prediction_df.columns if col.startswith('prob_')]
    
    # Создать фигуру
    plt.figure(figsize=(10, 8))
    
    # Получить уникальные томограммы
    tomos = set(ground_truth_df['tomo'].unique())
    prediction_df = prediction_df[prediction_df['tomo'].isin(tomos)]
    
    # Построить кривую ROC для каждого класса
    for class_name in class_names:
        # Инициализировать массивы для истинных меток и предсказаний
        y_true_all = []
        y_pred_all = []
        
        # Обработать каждую томограмму
        for tomo in tomos:
            # Получить опорные точки (истинные метки)
            gt_mask = (ground_truth_df['tomo'] == tomo) & (ground_truth_df[f'prob_{class_name}'] > 0.5)
            reference_points = ground_truth_df.loc[gt_mask, ['x', 'y', 'z']].values
            
            # Получить точки-кандидаты (предсказания)
            pred_mask = (prediction_df['tomo'] == tomo)
            candidate_points = prediction_df.loc[pred_mask, ['x', 'y', 'z']].values
            candidate_probs = prediction_df.loc[pred_mask, f'prob_{class_name}'].values
            
            if len(reference_points) == 0:
                continue
                
            if len(candidate_points) == 0:
                continue
            
            # Создать KDTree для опорных точек
            ref_tree = KDTree(reference_points)
            
            # Найти совпадения в пределах радиуса
            matches = ref_tree.query_ball_point(candidate_points, particle_radius[class_name])
            
            # Для каждой точки предсказания
            for i, match_indices in enumerate(matches):
                if match_indices:  # Если есть совпадение
                    y_true_all.append(1)  # Истинно-положительный
                else:
                    y_true_all.append(0)  # Ложно-положительный
                y_pred_all.append(candidate_probs[i])
        
        if not y_true_all:  # Пропустить, если точки не были обработаны
            continue
            
        # Вычислить кривую ROC
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
        roc_auc = auc(fpr, tpr)
        
        # Построить кривую ROC
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Добавить диагональную линию
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Настроить график
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Доля ложно-положительных')
    plt.ylabel('Доля истинно-положительных')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    return plt.gcf()

def plot_roc_curves_comparison(model_predictions, ground_truth_df, distance_multiplier=1.0, exclude_beta_amylase=True):
    """
    Plot ROC curves for each class comparing multiple models against ground truth.
    Uses KDTree to match points within particle radius, similar to compute_fbeta_score.
    
    Parameters:
    -----------
    model_predictions : dict
        Dictionary mapping model names to their prediction dataframes
    ground_truth_df : pandas.DataFrame
        DataFrame with ground truth probabilities
    title : str
        Title for the plot
    distance_multiplier : float
        Multiplier for particle radius thresholds
    exclude_beta_amylase : bool
        If True, beta_amylase class will be excluded from the plots
    """
    # Define particle radii
    particle_radius = {
        'apo_ferritin': 60,
        'beta_amylase': 65,
        'beta_galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus_like_particle': 135,
    }
    
    # Apply distance multiplier to radii
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}
    
    # Get class names from probability columns
    class_names = [col.replace('prob_', '') for col in ground_truth_df.columns if col.startswith('prob_')]
    
    # Filter out beta_amylase if requested
    if exclude_beta_amylase:
        class_names = [name for name in class_names if name != 'beta_amylase']
    
    # Create subplots for each class
    n_classes = len(class_names)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get unique tomograms
    tomos = set(ground_truth_df['tomo'].unique())
    
    # Plot ROC curve for each class
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        
        # Plot ROC curve for each model
        for model_name, pred_df in model_predictions.items():
            # Filter predictions to matching tomograms
            pred_df = pred_df[pred_df['tomo'].isin(tomos)]
            
            # Initialize arrays for true labels and predictions
            y_true_all = []
            y_pred_all = []
            
            # Process each tomogram
            for tomo in tomos:
                # Get reference points (ground truth)
                gt_mask = (ground_truth_df['tomo'] == tomo) & (ground_truth_df[f'prob_{class_name}'] > 0.5)
                reference_points = ground_truth_df.loc[gt_mask, ['x', 'y', 'z']].values
                
                # Get candidate points (predictions)
                pred_mask = (pred_df['tomo'] == tomo)
                candidate_points = pred_df.loc[pred_mask, ['x', 'y', 'z']].values
                candidate_probs = pred_df.loc[pred_mask, f'prob_{class_name}'].values
                
                if len(reference_points) == 0:
                    continue
                    
                if len(candidate_points) == 0:
                    continue
                
                # Create KDTree for reference points
                ref_tree = KDTree(reference_points)
                
                # Find matches within radius
                matches = ref_tree.query_ball_point(candidate_points, particle_radius[class_name])
                
                # For each prediction point
                for i, match_indices in enumerate(matches):
                    if match_indices:  # If there's a match
                        y_true_all.append(1)  # True positive
                    else:
                        y_true_all.append(0)  # False positive
                    y_pred_all.append(candidate_probs[i])
            
            if not y_true_all:  # Skip if no points were processed
                continue
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--')
        
        # Customize subplot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name}')
        ax.legend(loc="lower right")
        ax.grid(True)
    
    # Remove any unused subplots
    for idx in range(len(class_names), len(axes)):
        fig.delaxes(axes[idx])
        
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_roc_curves_by_model(model_predictions, ground_truth_df, distance_multiplier=1.0, exclude_beta_amylase=True):
    """
    Plot ROC curves for each model comparing all classes against ground truth.
    Uses KDTree to match points within particle radius, similar to compute_fbeta_score.
    
    Parameters:
    -----------
    model_predictions : dict
        Dictionary mapping model names to their prediction dataframes
    ground_truth_df : pandas.DataFrame
        DataFrame with ground truth probabilities
    title : str
        Title for the plot
    distance_multiplier : float
        Multiplier for particle radius thresholds
    exclude_beta_amylase : bool
        If True, beta_amylase class will be excluded from the plots
        
    Returns:
    --------
    tuple
        (fig, auc_df) where:
        - fig: matplotlib Figure with ROC curves for each model
        - auc_df: pandas DataFrame with AUC scores for each model and class
    """
    particle_radius = {
        'apo_ferritin': 60,
        'beta_amylase': 65,
        'beta_galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus_like_particle': 135,
    }
    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}
    
    class_names = [col.replace('prob_', '') for col in ground_truth_df.columns if col.startswith('prob_')]
    
    if exclude_beta_amylase:
        class_names = [name for name in class_names if name != 'beta_amylase']
    
    n_models = len(model_predictions)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    tomos = set(ground_truth_df['tomo'].unique())
    
    auc_scores = {}
    
    for idx, (model_name, pred_df) in enumerate(model_predictions.items()):
        ax = axes[idx]
        
        pred_df = pred_df[pred_df['tomo'].isin(tomos)]
        
        auc_scores[model_name] = {}
        
        for class_name in class_names:
            y_true_all = []
            y_pred_all = []
            
            for tomo in tomos:
                gt_mask = (ground_truth_df['tomo'] == tomo) & (ground_truth_df[f'prob_{class_name}'] > 0.5)
                reference_points = ground_truth_df.loc[gt_mask, ['x', 'y', 'z']].values
                
                pred_mask = (pred_df['tomo'] == tomo)
                candidate_points = pred_df.loc[pred_mask, ['x', 'y', 'z']].values
                candidate_probs = pred_df.loc[pred_mask, f'prob_{class_name}'].values
                
                if len(reference_points) == 0:
                    continue
                    
                if len(candidate_points) == 0:
                    continue
                
                ref_tree = KDTree(reference_points)
                
                matches = ref_tree.query_ball_point(candidate_points, particle_radius[class_name])
                
                for i, match_indices in enumerate(matches):
                    if match_indices:
                        y_true_all.append(1)
                    else:
                        y_true_all.append(0)
                    y_pred_all.append(candidate_probs[i])
            
            if not y_true_all:
                continue
                
            fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
            roc_auc = auc(fpr, tpr)
            
            auc_scores[model_name][class_name] = roc_auc
            
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name}')
        ax.legend(loc="lower right")
        ax.grid(True)
    
    # Remove any unused subplots
    for idx in range(len(model_predictions), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert AUC scores dictionary to DataFrame
    auc_df = pd.DataFrame(auc_scores).T
    
    # Round AUC scores to 4 decimal places
    auc_df = auc_df.round(4)
    
    return fig, auc_df

def plot_tomogram_slices_with_annotations(
    experiment_id,
    data_root_dir,
    z_slices_to_plot=None,
    ground_truth_df=None,
    prediction_dfs=None,
    model_names=None,
    classes_to_plot=None,
    particle_radius_angstrom=None,
    voxel_spacing=10.0,
    probability_threshold=0.5,
    prediction_marker_alpha=0.7,
    fixed_marker_size=8
):
    """
    Строит указанные 2D XY-срезы 3D томограммы из файла Zarr,
    и накладывает аннотации истинных меток и предсказаний.

    Параметры:
    -----------
    experiment_id : str
        ID эксперимента (например, "TS_99_9").
    data_root_dir : str
        Корневая директория, где хранятся данные эксперимента.
    z_slices_to_plot : list of int, optional
        Список индексов по оси Z для XY-срезов, которые нужно построить.
        Если None, будут выбраны 3 равномерно распределенных среза.
    ground_truth_df : pandas.DataFrame, optional
        DataFrame с аннотациями истинных меток. Ожидаемые колонки:
        'tomo', 'x', 'y', 'z', 'prob_className1', ...
    prediction_dfs : list of pandas.DataFrame, optional
        Список DataFrames с аннотациями предсказаний, структура аналогична ground_truth_df.
    model_names : list of str, optional
        Список имен, соответствующих prediction_dfs, для легенды.
        Если None, будут использованы имена по умолчанию, например "Model 1".
    classes_to_plot : list of str, optional
        Список имен классов частиц для визуализации (например, ['ribosome', 'apo_ferritin']).
        Если None, по умолчанию используется предопределенный список общих классов.
    particle_radius_angstrom : dict, optional
        Словарь, сопоставляющий имена классов с их радиусами в ангстремах.
        Если None, будут использованы радиусы по умолчанию. Они делятся на voxel_spacing.
    voxel_spacing : float, optional
        Расстояние между вокселями в ангстремах на воксель (по умолчанию: 10.0).
    probability_threshold : float, optional
        Минимальная вероятность для отображения аннотации (по умолчанию: 0.5).
    prediction_marker_alpha : float, optional
        Прозрачность альфа для маркеров предсказаний (по умолчанию: 0.7).
    fixed_marker_size : int, optional
        Фиксированный размер для всех маркеров (по умолчанию: 8).

    Возвращает:
    --------
    matplotlib.figure.Figure or None
        Объект фигуры, содержащий графики, или None, если произошла ошибка.
    """
    zarr_path = os.path.join(data_root_dir, experiment_id, f"VoxelSpacing{voxel_spacing:.3f}", "denoised.zarr")

    if not os.path.exists(zarr_path):
        print(f"Ошибка: Файл Zarr не найден по пути {zarr_path}")
        return None

    try:
        zf = zarr.open(zarr_path, mode='r')
        tomogram = np.array(zf['0']).transpose(2, 1, 0)
    except Exception as e:
        print(f"Ошибка загрузки томограммы из {zarr_path}: {e}")
        return None

    nx, ny, nz = tomogram.shape
    # print(f"Tomogram shape (original HWD, transposed to DHW): {tomogram.shape}")


    # Обработать z_slices_to_plot по умолчанию
    if z_slices_to_plot is None or not z_slices_to_plot:
        num_default_slices = min(3, nz)
        if nz > 1:
            z_slices_to_plot = np.linspace(0, nz - 1, num_default_slices, dtype=int).tolist()
        else:
            z_slices_to_plot = [0]
    z_slices_to_plot = [int(s) for s in z_slices_to_plot if 0 <= int(s) < nz]
    if not z_slices_to_plot:
        print(f"Предупреждение: Нет допустимых Z-срезов для построения томограммы с глубиной {nz}.")
        return None


    DEFAULT_CLASSES = ['apo_ferritin', 'beta_galactosidase', 'ribosome', 'thyroglobulin', 'virus_like_particle']
    if classes_to_plot is None:
        classes_to_plot = DEFAULT_CLASSES
    
    DEFAULT_RADII_ANGSTROM = {'apo_ferritin': 60, 'beta_amylase': 65, 'beta_galactosidase': 90, 'ribosome': 150, 'thyroglobulin': 130, 'virus_like_particle': 135, 'default': 50} # Добавлено по умолчанию
    if particle_radius_angstrom is None:
        particle_radius_angstrom = DEFAULT_RADII_ANGSTROM
    
    particle_radius_voxel = {k: v / voxel_spacing for k, v in particle_radius_angstrom.items()}

    if prediction_dfs:
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(prediction_dfs))]
        elif len(model_names) != len(prediction_dfs):
            print("Предупреждение: Длина model_names не соответствует длине prediction_dfs. Используются имена по умолчанию.")
            model_names = [f"Model {i+1}" for i in range(len(prediction_dfs))]
    else:
        prediction_dfs = [] # Убедиться, что это пустой список
        model_names = []

    # Определить цвета и стили
    if len(classes_to_plot) <= 10:
        color_palette = plt.cm.get_cmap('tab10').colors
    else:
        # Для большего числа классов рассмотреть более разнообразную цветовую карту или обеспечить достаточное количество цветов
        color_palette = plt.cm.get_cmap('tab20').colors # или plt.cm.get_cmap('hsv', len(classes_to_plot))

    class_colors = {cls_name: color_palette[i % len(color_palette)] for i, cls_name in enumerate(classes_to_plot)}

    # Определить формы маркеров
    gt_marker_shape = 'x'
    gt_plot_color = 'black' # Фиксированный цвет для всех маркеров GT
    prediction_marker_shapes = ['^', 's', 'P', '*', 'D', 'v', '<', '>'] 

    # Определить отдельную цветовую палитру для моделей
    if len(model_names) <= 10:
        model_color_palette = plt.cm.get_cmap('Dark2').colors # Палитра, хорошая для различий
    else:
        model_color_palette = plt.cm.get_cmap('gist_rainbow', len(model_names)).colors
    
    model_colors_map = {name: model_color_palette[i % len(model_color_palette)] for i, name in enumerate(model_names)}

    all_figures = [] # Список для хранения фигур для каждого Z-среза

    if not z_slices_to_plot or not classes_to_plot:
        print("Предупреждение: Нет Z-срезов или классов для построения.")
        return []

    for slice_idx_z in z_slices_to_plot:
        num_class_subplots = len(classes_to_plot)
        if num_class_subplots == 0:
            continue

        # Определить макет сетки для подграфиков классов внутри этой фигуры Z-среза
        n_cols_per_fig = int(np.ceil(np.sqrt(num_class_subplots)))
        n_rows_per_fig = int(np.ceil(num_class_subplots / n_cols_per_fig))

        current_fig, current_axes_array = plt.subplots(n_rows_per_fig, n_cols_per_fig, 
                                                     figsize=(n_cols_per_fig * 5, n_rows_per_fig * 5), 
                                                     squeeze=False)
        current_axes_flat = current_axes_array.flatten() # Сделать плоским для удобной итерации
        legend_handles_for_fig = {} # Сбросить дескрипторы легенды для каждой новой фигуры

        for class_subplot_idx, class_name in enumerate(classes_to_plot):
            if class_subplot_idx >= len(current_axes_flat): 
                break
            
            ax = current_axes_flat[class_subplot_idx]
            
            ax.imshow(tomogram[:, :, slice_idx_z].T, cmap='gray', origin='lower', aspect='equal')
            ax.set_title(f"Комплекс: {class_name}", fontsize=10)
            ax.set_xlabel("X (воксели)", fontsize=8)
            ax.set_ylabel("Y (воксели)", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)

            current_class_color = class_colors.get(class_name, 'gray')
            radius_vox = particle_radius_voxel.get(class_name, particle_radius_voxel.get('default', 50/voxel_spacing))

            # Построить истинные метки для этого класса и среза
            if ground_truth_df is not None:
                prob_col_gt = f'prob_{class_name}'
                if prob_col_gt in ground_truth_df.columns:
                    gt_for_tomo_class = ground_truth_df[
                        (ground_truth_df['tomo'] == experiment_id) &
                        (ground_truth_df[prob_col_gt] >= probability_threshold)
                    ]
                    for _, row in gt_for_tomo_class.iterrows():
                        x_vox, y_vox, z_vox = row['x']/voxel_spacing, row['y']/voxel_spacing, row['z']/voxel_spacing
                        if abs(z_vox - slice_idx_z) < 1.5:
                            ax.plot(x_vox, y_vox, 
                                    marker=gt_marker_shape, 
                                    markersize=fixed_marker_size, 
                                    color=gt_plot_color,
                                    linestyle='none', 
                                    alpha=0.7)
                    if 'Ground Truth' not in legend_handles_for_fig:
                        legend_handles_for_fig['Ground Truth'] = plt.Line2D(
                            [0], [0], 
                            marker=gt_marker_shape, color=gt_plot_color,
                            linestyle='none', markersize=7, label='Ground Truth'
                        )

            # Построить предсказания для этого класса и среза
            for model_idx, pred_df in enumerate(prediction_dfs):
                model_name_str = model_names[model_idx]
                model_specific_color = model_colors_map.get(model_name_str, 'black')
                model_specific_marker = prediction_marker_shapes[model_idx % len(prediction_marker_shapes)]

                prob_col_pred = f'prob_{class_name}'
                if prob_col_pred in pred_df.columns:
                    pred_for_tomo_class = pred_df[
                        (pred_df['tomo'] == experiment_id) &
                        (pred_df[prob_col_pred] >= probability_threshold)
                    ]
                    for _, row in pred_for_tomo_class.iterrows():
                        x_vox, y_vox, z_vox = row['x']/voxel_spacing, row['y']/voxel_spacing, row['z']/voxel_spacing
                        if abs(z_vox - slice_idx_z) < 1.5:
                            face_color_with_alpha_list = list(model_specific_color) 
                            if len(face_color_with_alpha_list) == 3: face_color_with_alpha_list.append(prediction_marker_alpha)
                            elif len(face_color_with_alpha_list) == 4: face_color_with_alpha_list[3] = prediction_marker_alpha
                            
                            ax.plot(x_vox, y_vox,
                                    marker=model_specific_marker, 
                                    markersize=fixed_marker_size, # Использовать фиксированный размер
                                    markerfacecolor=tuple(face_color_with_alpha_list),
                                    markeredgecolor=model_specific_color, 
                                    markeredgewidth=0.5, 
                                    linestyle='none')
                    if model_name_str not in legend_handles_for_fig:
                        legend_handles_for_fig[model_name_str] = plt.Line2D(
                            [0], [0], 
                            marker=model_specific_marker, 
                            markerfacecolor=model_specific_color, # Сплошной цвет в легенде
                            markeredgecolor=model_specific_color,
                            color=model_specific_color, # Резервный для некоторых маркеров, если mfc не виден
                            linestyle='none', markersize=7, label=model_name_str
                        )
        
        # Очистить неиспользуемые подграфики в текущей фигуре
        for i in range(num_class_subplots, len(current_axes_flat)):
            current_fig.delaxes(current_axes_flat[i])

        # Добавить легенду к текущей фигуре
        if legend_handles_for_fig:
            current_fig.legend(handles=list(legend_handles_for_fig.values()), loc='lower center', 
                               bbox_to_anchor=(0.5, 0.01), 
                               ncol=min(len(legend_handles_for_fig), 4), 
                               fontsize='medium')

        # Оценить нижнее поле, необходимое для легенды
        bottom_margin_fig = 0.05
        if legend_handles_for_fig:
            # Грубая оценка: 0.05 base + 0.03 per row of legend items (assuming up to 4 items per row)
            num_legend_rows = (len(legend_handles_for_fig) - 1) // min(len(legend_handles_for_fig),4) + 1
            bottom_margin_fig = 0.05 + num_legend_rows * 0.035
        
        current_fig.tight_layout(rect=[0, bottom_margin_fig, 1, 0.95]) # rect=[left, bottom, right, top]
        all_figures.append(current_fig)

    return all_figures

