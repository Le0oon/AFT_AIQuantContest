import sys
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import lightgbm as lgb
import argparse
import warnings
warnings.filterwarnings('ignore')
backtest_frame_path = '/home/liuqize/QuantFrame'
sys.path.append(backtest_frame_path)
try:
    from MLMethods import MLData, MLTrain, LGBM_EvalMetrics
except:
    error_msg = 'Please check the path of backtest_frame'
    print(error_msg)

data_path = '/data/user_home/liuqize/aft_data'
key_col = ['date', 'time', 'sym']
target_cols = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
data_path = '/data/user_home/liuqize/aft_data'
df = pd.read_pickle(os.path.join(data_path, 'all_data.pkl'))
features_cols = df.columns[3:-5].tolist()
feature2num = {col: i for i, col in enumerate(features_cols)}
num2feature = {i: col for i, col in enumerate(features_cols)}

new_feature_cols = list(range(len(features_cols)))
df.rename(columns=feature2num, inplace=True)
df['time_pk'] = df['date'].astype(str).str.cat(df['time'], sep='_')
train_df = df.query('date < 50')
test_df = df.query('date >= 50')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='label_5')
    parser.add_argument('--n_trials', type=int, default=1000)
    # parse args
    args = parser.parse_args()
    target = args.target
    print('Target:', target)
    print('Splitting Data...')
    SAMPLE_FRACTION = 0.1
    lab_df_list = []
    for lab in train_df[target].unique():
        lab_df_list.append(train_df.query(f'{target} == {lab}'))

    expected_smaple_num = int(len(train_df) * SAMPLE_FRACTION)

    cv_df = pd.concat([
        df.sample(expected_smaple_num // 3, random_state=42) for df in lab_df_list
    ])
    cv_df.reset_index(drop=True, inplace=True)

    # start validation
    index_list = MLData.ts_KFold_equal_time(cv_df, time_col='time_pk', asset_pk_col='sym', nsplit=5)
    # create study
    study = MLTrain.create_optuna_study('tuning_'+target, save_path='./optuna_studies/', direction='maximize')

    params = {
        'objective': 'multiclass',  # 多分类任务
        'num_class': 3,  # 类别数，这里是三分类任务
        'metric': 'multi_logloss',  # 多分类损失函数
        'n_jobs': cpu_count()//2,
        'force_col_wise': True,
        'verbose': -1,
    }
    print('Start Tuning...')
    MLTrain.LGBMSearchCV_manual(
        cv_df,
        x_cols=new_feature_cols,
        y_col=target,
        params=params,
        cv=index_list,
        feval=[LGBM_EvalMetrics.macro_f1],
        feval_name='macro_f1',
        study=study,
        n_trial=args.n_trials,

    )
