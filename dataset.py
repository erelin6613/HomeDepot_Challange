import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class HomeDepotDataset(Dataset):

	def __init__(self, product_id, attrs_frame):
		super(HomeDepotDataset, self).__init__()
		self.product_id = product_id
		self.attrs_frame = attrs_frame

	def __len__():
		return 

	def __getitem__():
		return

	def get_info(self):
		return self.attrs_frame.loc[self.attrs_frame['product_uid']==self.product_id]


if __name__ == '__main__':
	attrs_frame = pd.read_csv('attributes.csv')
	dataset = HomeDepotDataset(100001, attrs_frame)
	print(dataset.get_info())