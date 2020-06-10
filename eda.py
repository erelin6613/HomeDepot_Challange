import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from string import punctuation

tokenizer = TweetTokenizer()
stemmer = Stemmer('english')
word2vec_path = 'word2vec-homedepot.wv'

def explore_df(df):
	print(df)
	print('shape:', df.shape)
	print('stats:\n', df.describe())
	print('correlation:\n', df.corr())

def tokenize(string):
	t = [w for w in tokenizer.tokenize(string) 
	if w not in punctuation]
	return stemmer.stemWords(t)

def train_word2vec(texts, word2vec_path=None):
	if word2vec_path is None:
		print('Training Word2Vec...')
		vec = Word2Vec(texts)
		word2vec_path = 'word2vec-homedepot.wv'
		vec.save(word2vec_path)
		return (vec, word2vec_path)
	else:
		return (Word2Vec.load(word2vec_path), word2vec_path)

def preprocess_df(df, cols=['product_title', 'search_term', 'product_description']):
	for col in cols:
		print('Encoding', col)
		df.loc[:, col+'_tokenized'] = df[col].apply(tokenize)
		df = df.drop(col, axis=1)
	return df

def append_lists(l, l2):
	for el in l2:
		l.append(el)
	del l2
	return l


def main():
	train_df = pd.read_csv('train.csv', encoding='latin-1')
	test_df = pd.read_csv('test.csv', encoding='latin-1')
	word2vec_path = None
	desc_df = pd.read_csv('product_descriptions.csv', encoding='latin-1')
	train_df = pd.merge(train_df, desc_df, on='product_uid', how='left')
	del desc_df
	for group, frame in train_df.groupby('product_uid'):
		print('group:', group, '\n', frame)
		break
	train_df = preprocess_df(train_df)
	test_df = preprocess_df(test_df, cols=['product_title', 'search_term'])
	texts = []
	texts = append_lists(texts, train_df['product_title_tokenized'].tolist())
	texts = append_lists(texts, train_df['search_term_tokenized'].tolist())
	texts = append_lists(texts, train_df['product_description_tokenized'].tolist())
	texts = append_lists(texts, test_df['product_title_tokenized'].tolist())
	texts = append_lists(texts, test_df['search_term_tokenized'].tolist())
	with open('tokens.txt', 'a') as f:
		for t in texts:
			for w in t:
				f.write(w)
	del texts


if __name__ == '__main__':
	main()