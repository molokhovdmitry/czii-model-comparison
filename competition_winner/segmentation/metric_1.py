import numpy as np
import torch
from sklearn.metrics import roc_auc_score

"""
Получено из:
https://github.com/cellcanvas/album-catalog/blob/main/solutions/copick/compare-picks/solution.py
"""

import numpy as np
import pandas as pd

from scipy.spatial import KDTree


class ParticipantVisibleError(Exception):
    pass


def compute_metrics(reference_points, reference_radius, candidate_points):
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
    # Предотвращает отправку нескольких совпадений на одну частицу.
    # Это не будет строго корректным в (крайне редком) случае, когда истинные частицы
    # находятся очень близко друг к другу.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def calculate_f_scores(precision, recall, beta=1.0):
    """
    Рассчитывает метрики F1 и F-beta на основе значений точности и полноты.
    
    Аргументы:
        precision: Значение точности
        recall: Значение полноты
        beta: Параметр бета для метрики F-beta (по умолчанию 1.0, что дает F1)
        
    Возвращает:
        Кортеж (f1_score, f_beta_score)
    """
    if (precision + recall) == 0:
        return 0.0, 0.0
        
    # Рассчитать F1 (среднее гармоническое точности и полноты)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Рассчитать F-beta
    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1_score, f_beta_score


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int,
        weighted=True,
) -> float:
    '''
    F_beta и F1
      - истинно-положительный результат происходит, когда
         - (a) предсказанное местоположение находится в пределах порогового значения радиуса частицы, и
         - (b) указан правильный `particle_type`
      - необработанные результаты (TP, FP, FN) агрегируются по всем экспериментам для каждого типа частиц
      - f_beta и f1 рассчитываются для каждого типа частиц
      - отдельные оценки взвешиваются по типу частиц для итоговой оценки
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Фильтровать submission, чтобы включать только эксперименты, найденные в solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Разрешать только известные типы частиц
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    fbetas = []
    f1s = []
    fbeta_weights = []
    particle_types = []
    particle_details = {}
    
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        f1, fbeta = calculate_f_scores(precision, recall, beta=beta)
        
        fbetas.append(fbeta)
        f1s.append(f1)
        fbeta_weights.append(weights.get(particle_type, 1.0))
        particle_types.append(particle_type)
        
        particle_details[particle_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fbeta': fbeta
        }
        
    if weighted:
        aggregate_fbeta = np.average(fbetas, weights=fbeta_weights)
        aggregate_f1 = np.average(f1s, weights=fbeta_weights)
    else:
        aggregate_fbeta = np.mean(fbetas)
        aggregate_f1 = np.mean(f1s)
    
    metrics = {
        'fbeta': aggregate_fbeta,
        'f1': aggregate_f1
    }
    
    fbeta_dict = dict(zip(particle_types, fbetas))
    f1_dict = dict(zip(particle_types, f1s))
    
    return metrics, {'fbeta': fbeta_dict, 'f1': f1_dict, 'details': particle_details}

def calc_metric(cfg, pp_out, val_df, pre="val"):
    
    particles = cfg.classes
    pred_df = pp_out
    
    solution = val_df.copy()
    solution['id'] = range(len(solution))
    
    submission = pred_df.copy()
    submission['experiment'] = solution['experiment'].unique()[0]
    submission['id'] = range(len(submission))

    # Найти оптимальные пороговые значения для метрик F-beta и F1
    best_ths_fbeta = []
    best_ths_f1 = []
    for p in particles:
        sol0a = solution[solution['particle_type']==p].copy()
        sub0a = submission[submission['particle_type']==p].copy()
        fbeta_scores = []
        f1_scores = []
        ths = np.arange(0,0.5,0.005)
        for c in ths:
            metrics, _ = score(
                    sol0a.copy(),
                    sub0a[sub0a['conf']>c].copy(),
                    row_id_column_name = 'id',
                    distance_multiplier=0.5,
                    beta=4,
                    weighted = False)
            fbeta_scores.append(metrics['fbeta'])
            f1_scores.append(metrics['f1'])
            
        best_th_fbeta = ths[np.argmax(fbeta_scores)]
        best_th_f1 = ths[np.argmax(f1_scores)]
        best_ths_fbeta.append(best_th_fbeta)
        best_ths_f1.append(best_th_f1)
    
    # Создать submissions, используя оптимальные пороговые значения для обеих метрик
    submission_pp_fbeta = []
    submission_pp_f1 = []
    for th_fbeta, th_f1, p in zip(best_ths_fbeta, best_ths_f1, particles):
        submission_pp_fbeta.append(submission[(submission['particle_type']==p) & (submission['conf']>th_fbeta)].copy())
        submission_pp_f1.append(submission[(submission['particle_type']==p) & (submission['conf']>th_f1)].copy())
    
    submission_pp_fbeta = pd.concat(submission_pp_fbeta)
    submission_pp_f1 = pd.concat(submission_pp_f1)
    
    # Оценить результаты
    filtered_solution = solution[solution['particle_type']!='beta-amylase'].copy()
    
    fbeta_metrics, fbeta_details = score(
        filtered_solution,
        submission_pp_fbeta.copy(),
        row_id_column_name = 'id',
        distance_multiplier=0.5,
        beta=4)
    
    f1_metrics, f1_details = score(
        filtered_solution,
        submission_pp_f1.copy(),
        row_id_column_name = 'id',
        distance_multiplier=0.5,
        beta=1)  # Использовать beta=1 для метрики F1
    
    # Объединить результаты
    result = {}
    
    # Добавить оценки F-beta (метрика соревнования)
    for k, v in fbeta_details['fbeta'].items():
        result['score_' + k] = v
    result['score'] = fbeta_metrics['fbeta']
    
    # Добавить оценки F1
    for k, v in f1_details['f1'].items():
        result['f1_score_' + k] = v
    result['f1_score'] = f1_metrics['f1']
    
    return result

