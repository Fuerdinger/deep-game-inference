"""
DeepGameInference
Copyright (C) 2021 Daniel Fuerlinger

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np

class GameDataset(Dataset):
	def __init__(self, game_dict, keys, sizes):
		num_games = len(keys)
		self.sizes = sizes
		self.X_categorical = np.zeros((num_games, 3), dtype=np.int32)
		self.X_continuous = np.zeros((num_games, 3), dtype=np.float32)
		self.Y = np.zeros(num_games, dtype=np.float32)
		indices_to_remove = []
		
		self.num_items = 0
		i = 0
		for key in keys:
			current_game = game_dict[key]
			
			if current_game["metacritic"] == -1:
				current_game["metacritic"] = 70
			if len(current_game["genres"]) < 1 or current_game["gameplay_main"] == -1 or len(current_game["release_date"]) < 4:
				indices_to_remove.append(i)
				i = i + 1
				continue
			
			self.X_categorical[i][0] = current_game["series"]
			self.X_categorical[i][1] = current_game["genres"][0]
			self.X_categorical[i][2] = current_game["esrb"]
			
			self.X_continuous[i][0] = current_game["gameplay_main"]
			self.X_continuous[i][1] = current_game["metacritic"]
			self.X_continuous[i][2] = float((current_game["release_date"])[0:4])
			
			self.Y[i] = current_game["target_value"]
			self.num_items = self.num_items + 1
			i = i + 1
		
		self.X_categorical = np.delete(self.X_categorical, indices_to_remove, 0)
		self.X_continuous = np.delete(self.X_continuous, indices_to_remove, 0)
		self.Y = np.delete(self.Y, indices_to_remove)
	
	def __len__(self):
		return self.num_items
	
	def __getitem__(self, idx):
		series_tensor = torch.squeeze(F.one_hot(torch.tensor([self.X_categorical[idx][0]], dtype=torch.long), self.sizes[0])).type(torch.FloatTensor)
		genres_tensor = torch.squeeze(F.one_hot(torch.tensor([self.X_categorical[idx][1]], dtype=torch.long), self.sizes[1])).type(torch.FloatTensor)
		esrb_tensor = torch.squeeze(F.one_hot(torch.tensor([self.X_categorical[idx][2]], dtype=torch.long), self.sizes[2])).type(torch.FloatTensor)
		X_continuous = torch.from_numpy(self.X_continuous[idx])
		Y = self.Y[idx]
		return ((series_tensor, genres_tensor, esrb_tensor), X_continuous), Y

class TrainingModel(nn.Module):
	def __init__(self, sizes):
		super().__init__()
		self.lin = nn.Linear(6, 1)
		self.linCategorical = [nn.Linear(num,1) for num in sizes]
	
	def forward(self, xb):
		X_series = self.linCategorical[0](xb[0][0])
		X_genre = self.linCategorical[1](xb[0][1])
		X_esrb = self.linCategorical[2](xb[0][2])
		X_continuous = xb[1]
		x = torch.cat([X_series, X_genre, X_esrb, X_continuous], 1)
		x = torch.squeeze(self.lin(x))
		return x