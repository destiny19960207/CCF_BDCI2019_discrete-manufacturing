# CCF_BDCI2019_discrete-manufacturing
CCF_BDCI2019: Prediction of Quality Conformity Rate of Typical Workpieces in Discrete Manufacturing Process

params = {
'boosting_type': 'gbdt',
'objective': 'multiclassova',
'num_class': 4,  
'metric': 'multi_error', 
'num_leaves': 63,
'learning_rate': 0.01,
'feature_fraction': 0.9,
'bagging_fraction': 0.9,
'bagging_seed':0,
'bagging_freq': 1,
'verbose': -1,
'reg_alpha':1,
'reg_lambda':2,
'lambda_l1': 0,
'lambda_l2': 1,
'num_threads': 8,
}
lgb_train = lgb.Dataset(train_x, train_y)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1300,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=100,
                )
