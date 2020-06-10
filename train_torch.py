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


def train_model():
	epochs = 10
	X_1, X_2, y, _, _ = prep_main()
	model = HomeDepotModel()
	optimizer = Adam(model.parameters(), lr=0.001)
	for epoch in range(epochs):
		total_loss = []
		model.train()
		for sq, title, score in zip(X_1, X_2, y):
			model.zero_grad()
			out = model(torch.from_numpy(sq).float(), torch.from_numpy(title).float())
			y_tensor = torch.tensor(score[0]).view(1, ).long()
			loss = F.cross_entropy(out.float(), y_tensor)
			loss.backward()
			optimizer.step()
			total_loss.append(loss.item())
			optimizer.zero_grad()
		print('Epoch:', epoch, '\tLoss:', np.array(total_loss).mean())
	torch.save(model.state_dict(), 'model_weights.pth')
	return model

def get_predictions(model):
    _, _, _, X1, X2 = prep_main()
    preds = []
    for x1, x2 in zip(X1, X2):
        preds.append(model(x1, x2))
    return preds

if __name__ == '__main__':
	model = train_model()