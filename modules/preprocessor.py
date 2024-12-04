import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PATH = '../data/'
MODEL_PATH = '../models/'


# Определим функцию для выделения значений момента
def get_torque(torq_text):
    # Приведём текст к нижнему регистру
    torq_text = torq_text.lower()

    # Выделим значения с помощью регулярных выражений
    # Паттерн для выделения значений момента или скорости
    pattern_digits = r"[0-9,\.]{1,}"
    obtained_values = re.findall(pattern_digits, torq_text)

    # Паттерн для выделения единиц измерения
    pattern_units = r"[kgmrpn]{2,}"
    obtained_units = re.findall(pattern_units, torq_text)

    if len(obtained_units) == 0:
        return None
    elif len(obtained_units) == 1:
        if obtained_units[0] == 'rpm' or obtained_units[0] == 'nm':
            return float(obtained_values[0])
        else:
            # Приведём кгм к Нм
            return float(obtained_values[0]) * 9.80665
    elif len(obtained_units) == 2:
        if obtained_units[0] == 'nm':
            return float(obtained_values[0])
        else:
            # Приведём кгм к Нм
            return float(obtained_values[0]) * 9.80665
    else:
        return float(obtained_values[0])


# Определим функцию для выделения значений оборотов
def get_speed(torq_text):
    # Приведём текст к нижнему регистру
    torq_text = torq_text.lower()

    # Выделим значения с помощью регулярных выражений
    # Паттерн для выделения значений момента или скорости
    pattern_digits = r"[0-9,\.]{1,}"
    obtained_values = re.findall(pattern_digits, torq_text)

    # Паттерн для выделения единиц измерения
    pattern_units = r"[kgmrpn]{2,}"
    obtained_units = re.findall(pattern_units, torq_text)

    if len(obtained_units) in [0, 1, 2]:
        # Возвращаем значение скорости
        if len(obtained_values) == 2:
            return float(obtained_values[1].replace(',', ''))
        # Если два значения скорости, то вернём среднее
        if len(obtained_values) == 3:
            if float(obtained_values[2].replace(',', '')) > float(obtained_values[1].replace(',', '')):
                return 0.5 * (float(obtained_values[1].replace(',', '')) +
                              float(obtained_values[2].replace(',', '')))
            else:
                return float(obtained_values[1].replace(',', ''))
    else:
        return float(obtained_values[2].replace(',', ''))


# Определим трансформер, который удаляет единицы измерения
# из столбцов `mileage`, `engine`, `max_power`
class DelUnitsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        self.features = X_out.columns
        for col in self.features:
            X_out.loc[~X_out[col].isna(), col] = \
                X_out[~X_out[col].isna()][col].str.split().apply(
                    lambda x: float(x[0]) if len(x) == 2 else np.nan)
            X_out[col] = X_out[col].astype('float')
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Определим трансформер, который преобразует столбец с
# информацией о крутящем моменте в столбцы со значением скорости,
# при которой он достигается, и со значением момента
class SpeedMaxTorqTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Получим значение скорости
        X_out.loc[~X_out['torque'].isna(), 'max_torque_rpm'] = \
            X_out[~X_out['torque'].isna()]['torque'].apply(get_speed)

        # Получим значение момента
        X_out.loc[~X_out['torque'].isna(), 'torque'] = \
            X_out[~X_out['torque'].isna()]['torque'].apply(get_torque)
        X_out['torque'] = X_out['torque'].astype('float')
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Определим трансформер, который приводит к целочисленному типу
# передаваемые столбцы
class IntTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.features = X.columns
        return self

    def transform(self, X):
        X_out = X.copy().astype('int')
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Определим трансформер, который преобразует столбец с
# наименованием в столбцы с маркой и моделью
class NameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Получим марку автомобиля
        X_out['car_make'] = X_out['name'].str.split().apply(lambda x: x[0])
        # Получим значение момента
        X_out['car_model'] = X_out['name'].str.split().apply(lambda x: x[1])
        X_out.drop(['name'], axis=1, inplace=True)
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Определим трансформер, который добавляет столбец с
# пробегом в год
class KmTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Километраж в год
        X_out['km_per_year'] = X_out['km_driven'] / (2020 - X_out['year'] + 1)
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


# Определим трансформер, который добавляет столбец с
# максимальной удельной мощностью
class PwrTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Километраж в год
        X_out['relative_pwr'] = X_out['max_power'] / X_out['engine']
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features


class YearTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Километраж в год
        X_out['year_pow2'] = X_out['year'] ** 2
        self.features = X_out.columns
        return X_out

    def get_feature_names_out(self, X=None):
        return self.features
