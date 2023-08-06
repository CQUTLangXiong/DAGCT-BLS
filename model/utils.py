import yaml
import json
from sklearn.metrics import r2_score
import numpy as np
def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
def save_dict_to_json(dict_value: dict, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(dict_value, file, ensure_ascii=False)

def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


def mse(test_y,y_pred):
    error = test_y-y_pred
    return np.mean(np.array(error)**2)
def rmse(test_y,y_pred):
    error = test_y-y_pred
    return np.sqrt(np.mean(np.array(error)**2))
def mae(test_y,y_pred):
    error = np.abs(np.array(test_y-y_pred))
    return np.mean(error)
def mape(test_y,y_pred):
    error = np.abs(np.array((test_y-y_pred)/(test_y+1e-6)))
    return np.mean(error)
def r_2(test_y,y_pred):
    return r2_score(test_y,y_pred)
def get_all_result(test_y, y_pred):
    test_y = np.nan_to_num(test_y)
    y_pred = np.nan_to_num(y_pred)
    mse_day = mse(test_y,y_pred)
    rmse_day = rmse(test_y,y_pred)
    mae_day = mae(test_y,y_pred)
    mape_day = mape(test_y,y_pred)

    r2_day = r_2(np.asarray(test_y),np.asarray(y_pred))
    print(f'mse:{mse_day}, rmse:{rmse_day}, mae:{mae_day},mape:{mape_day},r2:{r2_day}')
    return mse_day, rmse_day, mae_day, mape_day, r2_day

def re_normalization(x, _mean, _std, _min, _max, scale_type='standard'):
    if scale_type == 'standard':
        x = x*_std + _mean
        return x
    else:
        x = x * (_max-_min)+_min
        return x