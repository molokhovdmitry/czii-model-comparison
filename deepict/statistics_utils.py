import numpy as np


def get_max_F1(F1_score: list):
    if len(F1_score) > 0:
        max_F1 = np.max(F1_score)
        optimal_peak_number = np.min(np.where(F1_score == max_F1)[0])
    else:
        max_F1 = 0
        optimal_peak_number = np.nan
    return max_F1, optimal_peak_number


def get_clean_points_close2point(point, clean, radius):
    close_to_point = []
    distances = []
    for clean_p in clean:
        dist = np.linalg.norm(clean_p - point)
        if dist <= radius:
            close_to_point.append(clean_p)
            distances.append(dist)
    close_to_point = [tuple(p) for p in close_to_point]
    return close_to_point, distances


def precision_recall_calculator(predicted_coordinates: np.array or list,
                                value_predicted: list,
                                true_coordinates: np.array or list,
                                radius: float,
                                particle_radii: dict = None,
                                particle_type: str = None):
    """
    Рассчитывает точность и полноту для обнаружения частиц.
    
    Аргументы:
        predicted_coordinates: Координаты предсказанных частиц
        value_predicted: Значения/оценки предсказанных частиц
        true_coordinates: Координаты истинных частиц
        radius: Радиус по умолчанию для определения соответствия двух координат одной и той же частице
        particle_radii: Словарь, сопоставляющий типы частиц с их радиусами в вокселях
        particle_type: Тип оцениваемой частицы
        
    Возвращает:
        Кортеж, содержащий точность, полноту и другие метрики
    """
    true_coordinates = list(true_coordinates)
    predicted_coordinates = list(predicted_coordinates)
    
    # Использовать критерий соревнования, если предоставлены радиусы частиц
    if particle_radii is not None and particle_type is not None and particle_type in particle_radii:
        # Частица считается "истинной", если она находится в пределах коэффициента 0.5 от радиуса интересующей частицы
        match_radius = 0.5 * particle_radii[particle_type]
        print(f"Используется радиус соответствия {match_radius} (0.5 × {particle_radii[particle_type]}) для {particle_type}")
    else:
        match_radius = radius
        print(f"Используется радиус соответствия по умолчанию {match_radius}")
    
    detected_true = list()
    predicted_true_positives = list()
    predicted_redundant = list()
    value_predicted_true_positives = list()
    value_predicted_redundant = list()
    precision = list()
    recall = list()
    total_true_points = len(true_coordinates)
    assert total_true_points > 0, "один пустой список здесь!"
    if len(predicted_coordinates) == 0:
        print("Нет предсказанных точек")
        precision = []
        recall = []
        detected_true = []
        predicted_true_positives = []
        predicted_false_positives = []
        value_predicted_true_positives = []
        value_predicted_false_positives = []
        predicted_redundant = []
        value_predicted_redundant = []
        false_negatives = total_true_points
    else:
        predicted_false_positives = list()
        value_predicted_false_positives = list()
        for value, point in zip(value_predicted, predicted_coordinates):
            close_to_point, distances = get_clean_points_close2point(
                point,
                true_coordinates,
                match_radius
            )
            if len(close_to_point) > 0:
                flag = "true_positive_candidate"
                flag_tmp = "not_redundant_yet"
                for dist, clean_p in sorted(zip(distances, close_to_point)):
                    if flag == "true_positive_candidate":
                        if tuple(clean_p) not in detected_true:
                            detected_true.append(tuple(clean_p))
                            flag = "true_positive"
                        else:
                            flag_tmp = "redundant_candidate"
                    # else:
                    # print(point, "is already flagged as true positive")
                if flag == "true_positive":
                    predicted_true_positives.append(tuple(point))
                    value_predicted_true_positives.append(value)
                elif flag == "true_positive_candidate" and \
                        flag_tmp == "redundant_candidate":
                    predicted_redundant.append(tuple(point))
                    value_predicted_redundant.append(value)
                else:
                    print("Этого никогда не должно произойти!")
            else:
                predicted_false_positives.append(tuple(point))
                value_predicted_false_positives.append(value)
            true_positives_total = len(predicted_true_positives)
            false_positives_total = len(predicted_false_positives)
            total_current_predicted_points = true_positives_total + \
                                             false_positives_total
            precision.append(true_positives_total / total_current_predicted_points)
            recall.append(true_positives_total)
        false_negatives = [point for point in true_coordinates if tuple(point) not in detected_true]
        N_inv = 1 / total_true_points
        recall = np.array(recall) * N_inv
        recall = list(recall)
    return precision, recall, detected_true, predicted_true_positives, \
           predicted_false_positives, value_predicted_true_positives, \
           value_predicted_false_positives, false_negatives, predicted_redundant, \
           value_predicted_redundant


def f1_score_calculator(precision: list, recall: list):
    f1_score = []
    if len(precision) == 0:
        print("Нет точности и полноты")
        f1_score = [0]
    else:
        for p, r in zip(precision, recall):
            if p + r != 0:
                f1_score.append(2 * p * r / float(p + r))
            else:
                f1_score.append(0)
    return f1_score


def f_beta_score_calculator(precision: list, recall: list, beta: float = 4.0):
    """
    Вычисляет оценку F-beta для значений точности и полноты.
    F-beta - это взвешенное гармоническое среднее точности и полноты,
    где beta контролирует вес полноты в комбинированной оценке.
    
    F-beta = (1 + beta²) * (precision * recall) / (beta² * precision + recall)
    
    Когда beta = 1, это эквивалентно F1-score.
    Когда beta > 1, больший вес придается полноте (beta=4 используется в соревновании).
    Когда beta < 1, больший вес придается точности.
    
    Аргументы:
        precision: Список значений точности
        recall: Список значений полноты
        beta: Значение beta для использования (по умолчанию: 4.0 согласно требованиям соревнования)
        
    Возвращает:
        Список оценок F-beta
    """
    f_beta_score = []
    if len(precision) == 0:
        print("Нет точности и полноты")
        f_beta_score = [0]
    else:
        beta_squared = beta * beta
        for p, r in zip(precision, recall):
            if p + r != 0:
                f_beta_score.append((1 + beta_squared) * p * r / float(beta_squared * p + r))
            else:
                f_beta_score.append(0)
    return f_beta_score


def get_max_F_beta(f_beta_score: list):
    """
    Получает максимальную оценку F-beta и соответствующий ей индекс.
    
    Аргументы:
        f_beta_score: Список оценок F-beta
        
    Возвращает:
        Кортеж (max_f_beta, optimal_peak_number)
    """
    if len(f_beta_score) > 0:
        max_f_beta = np.max(f_beta_score)
        optimal_peak_number = np.min(np.where(f_beta_score == max_f_beta)[0])
    else:
        max_f_beta = 0
        optimal_peak_number = np.nan
    return max_f_beta, optimal_peak_number


def quadrature_calculator(x_points: list, y_points: list) -> float:
    """
    Эта функция вычисляет приближенное значение интеграла реальной
    функции f в интервале, используя метод трапеций.

    Вход:
    x_points: список точек на оси x (не обязательно упорядоченный)
    y_points: список точек, таких что y_points[n] = f(x_points[n]) для
    каждого n.
    """
    # sorted_y = [p for _, p in sorted(zip(x_points, y_points))]
    sorted_y = [p for _, p in
                sorted(list(zip(x_points, y_points)), key=lambda x: x[0])]
    n = len(y_points)
    sorted_x = sorted(x_points)

    trapezoidal_rule = [
        0.5 * (sorted_x[n + 1] - sorted_x[n]) * (sorted_y[n + 1] + sorted_y[n])
        for n in range(n - 1)]

    return float(np.sum(trapezoidal_rule))


def pr_auc_score(precision: list, recall: list) -> float:
    """
    Эта функция вычисляет приближенное значение площади
    под кривой точность-полнота (PR).
    """
    return quadrature_calculator(recall, precision)
