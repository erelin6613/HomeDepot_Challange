import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import lightgbm as lgb

from preprocess import main as prep_main

def make_predictions(model, inputs):
	return model.predict(inputs)

def train(X, y, predict=False, to_predict=None):
	kf = RepeatedKFold(n_splits=5, n_repeats=2)
	model = lgb.LGBMClassifier()

	preds_tmp = []
	for train_index, test_index in kf.split(X):
		print('Fitting Fold...')
		X_train, X_test = X.values[train_index], X.values[test_index]
		y_train, y_test = y.values[train_index], y.values[test_index]
		model.fit(X_train, y_train)
		if predict:
			preds_tmp.append(np.exp(model_3.predict(to_predict)))
	if predict:
		preds = np.mean(np.array(preds_tmp), 0)
		return (model, preds)
	return model

def stack_training(cols=['product_title_tokenized_ids',
	'search_term_tokenized_ids'], targets=['relevance']):
	
	train_df, test_df = prep_main()
	models = []
	for col in cols:
		model_pt, train_df.loc[:, col] = train(train_df[col], 
			train_df[targets], True, train_df[col])
		test.loc[:, col] = make_predictions(model_pt)
	print(train_df)
	print(test_df)

if __name__ == '__main__':
	
	stack_training()