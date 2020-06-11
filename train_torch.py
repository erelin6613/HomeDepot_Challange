import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

# this is preprocessing script converting
# our data to numeric features
from preprocess import main as prep_main

class HomeDepotModel(nn.Module):
	def __init__(self, max_len_sq=5,
		max_len_title=15, 
		output_size=13, 
		hidden_size1=32, 
		hidden_size2=64):

		super().__init__()
		self.max_len_sq = max_len_sq
		self.max_len_title = max_len_title
		self.output_size = output_size
		self.lin_sq = nn.Linear(self.max_len_sq, hidden_size1)
		self.lin_title = nn.Linear(self.max_len_title, hidden_size1)
		self.lin2 = nn.Linear(hidden_size1*2, hidden_size2)
		self.lin3 = nn.Linear(hidden_size2, hidden_size1)
		self.lin4 = nn.Linear(hidden_size1, output_size)
		
	def forward(self, x1, x2):
		x1 = F.relu(self.lin_sq(x1))
		x2 = F.relu(self.lin_title(x2))
		x = torch.cat((x1, x2), 0)
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		x = F.softmax(self.lin4(x), dim=0)

		return x.view(1, self.output_size)

def generate_test_val(X1, X2, y, test_size=0.2):

	assert X1.shape[0] == X2.shape[0] == y.shape[0]
	test_len = int(y.shape[0]*test_size)

	ind = np.random.choice(y.shape[0], test_len)
	X1_val, X2_val, y_val = X1[ind], X2[ind], y[ind]
	X1_train = np.delete(X1, ind, 0)
	X2_train = np.delete(X2, ind, 0)
	y_train = np.delete(y, ind, 0)
	return X1_train, X2_train, y_train, X1_val, X2_val, y_val 

def correct(outputs, labels):
	prob = torch.softmax(outputs, dim=1)
	pred = prob.argmax(dim=1)
	correct = (pred == labels)
	if correct:
		return 1
	return 0

def train_model(model_save_path='model_weights.pth'):
	epochs = 5
	X_1, X_2, y, _, _, _ = prep_main()

	model = HomeDepotModel()
	optimizer = Adam(model.parameters(), lr=0.0001)
	X1_train, X2_train, y_train, X1_val, X2_val, y_val = generate_test_val(
		X_1, X_2, y)
	for epoch in range(epochs):
		pbar = tqdm(total=len(X1_train))
		total_loss = []
		val_loss = []
		val_acc = 0#
		model.train()
		for sq, title, score in zip(X1_train, X2_train, y_train):
			model.zero_grad()
			out = model(torch.from_numpy(sq).float(), torch.from_numpy(title).float())
			y_tensor = torch.tensor(score[0]).view(1, ).long()
			loss = F.cross_entropy(out.float(), y_tensor)
			loss.backward()
			optimizer.step()
			total_loss.append(loss.item())
			optimizer.zero_grad()
			pbar.update(1) 
		pbar.close()
		vbar = tqdm(total=len(X1_val))
		for sq, title, score in zip(X1_val, X2_val, y_val):

			out = model(torch.from_numpy(sq).float(), torch.from_numpy(title).float())
			y_tensor = torch.tensor(score[0]).view(1, ).long()
			val_loss.append(F.cross_entropy(out.float(), y_tensor).item())
			val_acc += correct(out, y_tensor)
			vbar.update(1)
		vbar.close()
		print('Epoch:', epoch, '\tLoss:', np.array(total_loss).mean(),
			'\tval_loss:', np.array(val_loss).mean(), 
			'\tval_acc:', val_acc/len(X1_val))
	torch.save(model.state_dict(), model_save_path)
	
	return model

def decode_ids(tokenizer, word):
	for k, v in tokenizer.word_index.items():
		if v==word:
			return k
	return ' '

def get_predictions(model):
	X1, X2, _, _, _, tokenizer = prep_main()
	model.eval()
	preds = []

	for x1, x2 in zip(X1, X2):
		preds.append(model(torch.from_numpy(x1).float(), torch.from_numpy(x2).float()))
	return X1, X2, preds, tokenizer

if __name__ == '__main__':
	train_model()