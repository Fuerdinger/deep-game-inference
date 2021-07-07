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

import json
import random
from model_dataset import TrainingModel
from model_dataset import GameDataset
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import argparse
import os

def get_model(sizes, lr):
	model = TrainingModel(sizes)
	return model, optim.SGD(model.parameters(), lr=lr)

# taken from https://pytorch.org/tutorials/beginner/nn_tutorial.html
def loss_batch(model, loss_func, xb, yb, opt=None):
	loss = loss_func(model(xb), yb)
	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()
	return loss.item(), len(xb)

# taken from https://pytorch.org/tutorials/beginner/nn_tutorial.html
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
	for epoch in range(epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, opt)
		
		model.eval()
		with torch.no_grad():
			losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
		print(epoch, val_loss)
		
# taken mostly from https://pytorch.org/tutorials/beginner/nn_tutorial.html
def get_data(train_ds, valid_ds, bs, vs):
	return (DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=vs))

def get_data_sets(games_dict, sizes):
	keys = list(games_dict)
	random.shuffle(keys)
	split_point = int(len(keys) * 0.8)
	train_ds = GameDataset(games_dict, keys[0:split_point], sizes)
	valid_ds = GameDataset(games_dict, keys[split_point:], sizes)
	return train_ds, valid_ds

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=False, help="name of the game data file to use for training", default="seeded_data/data.json")
	parser.add_argument("--output", required=False, help="name of the model to output", default="models/model.pt")
	parser.add_argument("--batch_size", required=False, help="batch size for training. Use smaller batches for smaller datasets!", default=32)
	parser.add_argument("--validity_size", required=False, help="size of the validity batch which loss will be calculated on", default=128)
	parser.add_argument("--epochs", required=False, help="the number of epochs to train for", default=1000)
	parser.add_argument("--learning_rate", required=False, help="the learning rate for training", default=0.000001)
	args = parser.parse_args()
	
	with open(args.input, "r", encoding='utf-8') as f:
		data_file = json.loads(f.read())
		games_json = data_file["games"]
		map_json = data_file["map"]["id-to-name"]
	
	num_series = len(map_json["series"].keys())
	num_genres = len(map_json["genres"].keys())
	num_esrb_ratings = len(map_json["esrb_ratings"])
	sizes = (num_series, num_genres, num_esrb_ratings)
	train_ds, valid_ds = get_data_sets(games_json, sizes)
	
	bs = int(args.batch_size)
	vs = int(args.validity_size)
	lr = float(args.learning_rate)  # learning rate
	epochs = int(args.epochs)  # how many epochs to train for
	train_dl, valid_dl = get_data(train_ds, valid_ds, bs, vs)
	model, opt = get_model(sizes, lr)
	fit(epochs, model, torch.nn.L1Loss(), opt, train_dl, valid_dl)
	
	print("Saving to " + args.output)
	if os.path.exists(args.output):
		os.remove(args.output)
	torch.save(model.state_dict(), args.output)