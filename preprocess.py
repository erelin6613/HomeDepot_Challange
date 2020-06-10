import os
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from string import punctuation

from sentencepiece_train import *

tokenizer = TweetTokenizer()

def tokenize(string):
	t = [w.lower() for w in tokenizer.tokenize(string) 
	if w not in punctuation]
	return t

def preprocess_df(df, cols=['product_title', 
	'search_term'],
	test=False):
	if not test:
		df = map_scores(df)
	for col in cols:
		print('Encoding', col)
		df.loc[:, col+'_tokenized'] = df[col].apply(tokenize)
		df = df.drop(col, axis=1)
	return df

def texts_to_numeric(df, 
	cols=['product_title_tokenized', 
	'search_term_tokenized']):
	for col in cols:
		print('Getting IDs:', col)
		df.loc[:, col+'_ids'] = df[col].apply(tokens_to_ids)
		df = df.drop(col, axis=1)
	return df

def append_lists(l, l2):
	for el in l2:
		l.append(el)
	del l2
	return l

def map_scores(df):
	scores = {1.00: 0, 1.25: 1, 1.33: 2, 1.5: 3,
		1.67: 4, 1.75: 5, 2.00: 6, 2.25: 7,
		2.33: 8, 2.5: 9, 2.67: 10, 
		2.75: 11, 3.00: 12}
	df.loc[:, 'relevance'] = df['relevance'].map(scores)
	return df

def cat_encoder(df, columns, target=None, test=False, encoder=None):
    if test:
        return encoder.transform(df)
    else:
        encoder = OneHotEncoder(cols=columns, return_df=True)
        encoder.fit(df, target)
        df = encoder.transform(df)
        return (df, encoder)

def pad_sequences(l, max_len):
	arr = np.array(l)
	if len(arr) > max_len:
		return arr[:max_len].astype(np.int64)
	elif len(arr) == max_len:
		return arr.astype(np.int64)
	else:
		mask = np.ones((max_len, ))*(-1)
		mask[:arr.shape[0]] = arr
		return mask.astype(np.int64)

def get_numpy_arrays(df, cols=['product_title', 
	'search_term'], target_col=None):
	X = (df['search_term_tokenized_ids'].values, 
		df['product_title_tokenized_ids'].values)
	X_1 = []
	X_2 = []
	for i in range(len(X[0])):
		X_1.append(np.array([int(x) for x in X[0][i]]))
	for i in range(len(X[1])):
		X_2.append(np.array([int(x) for x in X[1][i]]))
	X_1, X_2 = np.array(X_1), np.array(X_2)
	if target_col:
		y = np.array([[int(x)] for x in df[target_col].values])
		return (X_1, X_2, y)
	return (X_1, X_2)
	pass

def main():
	train_df = pd.read_csv('train.csv', encoding='latin-1')
	test_df = pd.read_csv('test.csv', encoding='latin-1')
	desc_df = pd.read_csv('product_descriptions.csv', encoding='latin-1')
	train_df = pd.merge(train_df, desc_df, on='product_uid', how='left')
	del desc_df
	train_df = preprocess_df(train_df)
	test_df = preprocess_df(test_df, 
		cols=['product_title', 'search_term'], test=True)

	# skip this step if you have text file with vocab
	if not os.path.isfile('tokens.txt'):
		texts = []
		texts = append_lists(texts, 
			train_df['product_title_tokenized'].tolist())
		texts = append_lists(texts, 
			train_df['search_term_tokenized'].tolist())
		texts = append_lists(texts, 
			train_df['product_description_tokenized'].tolist())
		texts = append_lists(texts, 
			test_df['product_title_tokenized'].tolist())
		texts = append_lists(texts, 
			test_df['search_term_tokenized'].tolist())
		with open('tokens.txt', 'a') as f:
			for t in texts:
				for w in t:
					f.write(w+'\n')
		del texts

	# skip this step if you have SentencePiece model
	if not os.path.isfile('m.model'):
		train_sp_model('tokens.txt')
	train_df = texts_to_numeric(train_df)
	test_df = texts_to_numeric(test_df, cols=['product_title_tokenized', 
		'search_term_tokenized'])
	train_df.loc[:, 
		'product_title_tokenized_ids'] = train_df['product_title_tokenized_ids'].apply(
		pad_sequences, args=(15, ))
	train_df.loc[:, 
		'search_term_tokenized_ids'] = train_df['search_term_tokenized_ids'].apply(
		pad_sequences, args=(5, ))
	train_df = train_df.drop('product_description', axis=1)
	X1_train, X2_train, y_train = get_numpy_arrays(train_df, target_col='relevance')
	X1_test, X2_test = get_numpy_arrays(test_df)

	return X1_train, X2_train, y_train, X1_test, X2_train

if __name__ == '__main__':
	main()