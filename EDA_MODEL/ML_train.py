# 라이브러리 임포트

import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import sklearn.metrics

import psutil
import time

from IPython.display import display
pd.options.display.max_columns = None

import warnings
warnings.filterwarnings('ignore')


# 폴더 만들기
def createDirectory(folder_path):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        print(f"Error: Failed to create {folder_path}")
        

# 데이터 로드
def load_data(data_path):
    if os.path.isfile(data_path):
        return pd.read_csv(str(data_path), index_col=0)
    else:
        print(f"{data_path}가 올바르지 않습니다!")
        
        
# 데이터 나누기
def data_split(train_path):
    
    # 데이터 가져오기
    df_train = load_data(train_path)
    
    target = df_train['GT']
    train = df_train.drop(['GT'], axis=1)
    
    X_train1, X_test, y_train1, y_test = train_test_split(train, target, test_size=0.1, stratify=target, random_state=107)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train1, y_train1, test_size=0.1, stratify=y_train1, random_state=107)
    
    column_names = X_train1.columns
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    scaler.fit(X_train1)
    X_train1_scaled = scaler.transform(X_train1)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train1_scaled, X_train_scaled, X_valid_scaled, X_test_scaled, y_train1, y_train, y_valid, y_test, column_names
        

# 훈련시간 측정 함수
def memory_usage(message: str = 'debug'):
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB")

    
# 변수 중요도 시각화 함수
def plot_feature_importance(importance, names, model_name):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #fi_df = fi_df.tail(50)
    
    #Define size of bar plot
    plt.figure(figsize=(30,30))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.ioff()
    #Add chart labels
    plt.title(model_name + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.close()
    plt.savefig(f'train_result_img/{model_name}_plot_feature_importance.jpg')
    
    
# 모델 선택
def model_selection(model_name, study):
    if model_name == 'LGBM':
        model = LGBMClassifier(**study.best_params)
    elif model_name == 'XGB':
        model = XGBClassifier(**study.best_params)
    elif model_name == 'CAT':
        model = CatBoostClassifier(**study.best_params)
    return model
    
    
# optuna 함수
def objective(trial: Trial, model_name, X_train, X_valid, y_train, y_valid):
    
    if model_name == 'LGBM':
        params_lgbm = {           
                        "random_state": 42,
                        "verbosity": -1,
                        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
                        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
                        "max_depth": trial.suggest_int("max_depth", 1, 20),
                        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
                        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                        "max_bin": trial.suggest_int("max_bin", 200, 500),
                        }
        model = LGBMClassifier(**params_lgbm)
        
        
    elif model_name == 'XGB':
        params_xgb = {
                        "random_state":42,
                        "objective": "multi:softprob",
                        "eval_metric":'mlogloss',
                        "booster": 'gbtree',
                        # 'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'gpu_id': 0, # GPU 사용시
                        "tree_method": 'exact', 'cpu_id': -1,  # CPU 사용시
                        "verbosity": 0,
                        'num_class':5,
                        "max_depth": trial.suggest_int("max_depth", 4, 10),
                        "learning_rate": trial.suggest_uniform('learning_rate', 0.0001, 0.99),
                        'n_estimators': trial.suggest_int("n_estimators", 1000, 10000, step=100),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
                        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 1),
                        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 1),
                        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),     
                        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),
                        "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
                        # 'num_parallel_tree': trial.suggest_int("num_parallel_tree", 1, 500) 추가하면 느려짐.
                        }
        model = XGBClassifier(**params_xgb)
        
        
    elif model_name == 'CAT':
        params_cat = {
                        "random_state":42,
                        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 50.00),
                        "n_estimators":trial.suggest_int("n_estimators", 1000, 10000),
                        "max_depth":trial.suggest_int("max_depth", 4, 10), # 10 넘으면 너무 느려진다...
                        'random_strength' :trial.suggest_int('random_strength', 0, 100),
                        "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
                        "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                        "max_bin": trial.suggest_int("max_bin", 200, 500),
                        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                        }
        model = CatBoostClassifier(**params_cat)
    
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)],
              early_stopping_rounds=100, verbose=False)
    
    pred = model.predict(X_valid)
    f1 = f1_score(y_valid, pred)
    print('f1: {0:.4f}'.format(f1))
    
    return f1

# 평가함수
def get_clf_eval(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    print('accuracy: {0:.4f}'.format(accuracy))
    print('f1: {0:.4f}'.format(f1))
    print('precision: {0:.4f}'.format(precision))
    print('recall(Sensitiviy): {0:.4f}'.format(recall))
    print('Specificity: ', round(cm[0][0]/(cm[0][0]+cm[0][1]) ,4))
    print('confusion_matrix: ')
    print(cm)

# 트레인 함수
def train_optuna(model_name, trial_num, train_path, threshold, stacking=False):
    
    X_train1, X_train, X_valid, X_test, y_train1, y_train, y_valid, y_test, column_names = data_split(train_path)
    
    createDirectory('train_result_img')
    createDirectory('result_csv')
    
    train_start = time.time()
    
    print('1.최적의 파라미터를 찾기 위한 훈련 시작')
    study = optuna.create_study(
    study_name=f"{model_name}_parameter_opt",
    direction="maximize",
    sampler=TPESampler(seed=42),
    )
    
    study.optimize(lambda trial: objective(trial, model_name, X_train, X_valid, y_train, y_valid), n_trials=trial_num)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    
    print('2.훈련 결과 시각화')
    # plotly.graph_objs._figure.Figure 저장하는 방법
    # https://plotly.com/python/static-image-export/
    # 시각화
    visualization.plot_optimization_history(study).write_image(f'train_result_img/{model_name}_plot_optimization_history.jpg')
    # 파라미터들관의 관계
    visualization.plot_parallel_coordinate(study).write_image(f'train_result_img/{model_name}_plot_parallel_coordinate.jpg')
    # 하이퍼파라미터 중요도
    visualization.plot_param_importances(study).write_image(f'train_result_img/{model_name}_plot_param_importances.jpg')
    
    if stacking:
        X_train, X_valid, y_train, y_valid =  train_target_split(train, target)
        print('스태킹 모델 훈련&추론 중')
        X_valid_model = model_selection(model_name, study)
        X_valid_model.fit(X_train, y_train)
        X_valid_pred = X_valid_model.predict(X_valid)
        test_pred = X_valid_model.predict(test)
        
        return  X_valid_pred, test_pred
    
    else:
        print('3.최적의 파라미터로 다시 학습')
        model = model_selection(model_name, study)
        
        model.fit(X_train1, y_train1)
        # 컬럼 중요도
        plot_feature_importance(model.feature_importances_, column_names, model_name) 
        
        print('4.학습된 모델 예측')
        pred = model.predict(X_test)
        
        print('5.모델 예측 결과')
        get_clf_eval(y_test, pred)
        
        print(f'6.임계값 {threshold}으로 모델 예측 결과')
        pred_proba = model.predict_proba(X_test)
        custom_threshold = threshold
        pred_proba_1 = pred_proba[:,1].reshape(-1,1)
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
        custom_predict = binarizer.transform(pred_proba_1)
        get_clf_eval(y_test, custom_predict)

        memory_usage("학습하는데 걸린 시간  {:.2f} 분\n".format( (time.time() - train_start)/60))
        
        return model
