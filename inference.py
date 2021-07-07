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
import torch
from model_dataset import TrainingModel
import argparse
import rawg.rawgpy as rawgpy
from howlongtobeatpy import HowLongToBeat
import torch.nn.functional as F


def get_property(map, property, name, fallback_to_none=False):
	if name not in map[property]:
		if fallback_to_none is False:
			print("Error: could not find \"" + name + "\" in map of \"" + property + "\" to ids. Try a different name, or use \"None\" if your training data has nothing for this category.")
			exit(-1)
		else:
			print("Could not find \"" + name + "\" in map of \"" + property + "\" to ids. Defaulting to \"None\"")
			return map[property]["None"]
	return map[property][name]

def get_attributes_from_user(args, names_to_ids):
	series = get_property(names_to_ids, "series", args.series)
	genre = get_property(names_to_ids, "genres", args.genre)
	esrb = get_property(names_to_ids, "esrb_ratings", args.esrb)
	gameplay_main = float(args.gameplay)
	metacritic = float(args.metacritic)
	release_date = float(args.release)
	return series, genre, esrb, gameplay_main, metacritic, release_date

def get_htlb_data(args, names_to_ids):
	gameplay_main = args.gameplay
	if gameplay_main != -1:
		print("Using user selected override for gameplay time")
		gameplay_main = float(gameplay_main)
	else:
		results = HowLongToBeat().search(args.name)
		if len(results) == 0:
			print("No HowLongtobeat results were found for \"" + args.name + "\", please try a different name or specify an override with --gameplay")
			exit(-1)
		if len(results) < args.selector:
			print("Only " + len(results) + " Howlongtobeat results were found, yet you selected with index " + args.selector + ". Please try again with a smaller index")
			exit(-1)
		game_hltb = results[args.selector]
		if game_hltb.gameplay_main == -1:
			print("No gameplay time found in Howlongtobeat page for the game. Please specify an override with --gameplay")
			exit(-1)
		if game_hltb.gameplay_main_unit != "Hours":
			gameplay_main = 1.
		else:
			gameplay_main = float(game_hltb.gameplay_main.replace("½", ""))
			
	return gameplay_main

def get_rawg_data(args, names_to_ids):
	with open("rawg_info.json", "r", encoding='utf-8') as f:
		rawg_data = json.loads(f.read())
		user_agent = rawg_data["user-agent"]
		api_key = rawg_data["key"]
	
	rawg = rawgpy.RAWG(user_agent, api_key)
	results = rawg.search(args.name)
	if len(results) == 0:
		print("No RAWG results were found for \"" + args.name + "\", please try a different name")
		exit(-1)
	if len(results) < args.selector:
		print("Only " + len(results) + " RAWG results were found, yet you selected with index " + args.selector + ". Please try again with a smaller index")
		exit(-1)
	game_rawg = results[args.selector]
	game_rawg.populate()
	
	genre = args.genre
	esrb = args.esrb
	metacritic = args.metacritic
	release_date = args.release
	
	if genre != "None":
		print("Using user override for genre")
		genre = get_property(names_to_ids, "genres", genre)
	else:
		if len(game_rawg.genres) == 0:
			print("No genre data found from RAWG. Please specify an override with --genre")
			exit(-1)
		else:
			genre = get_property(names_to_ids, "genres", game_rawg.genres[0].name, True)
			
	if esrb != "None":
		print("Using user override for esrb")
		esrb = get_property(names_to_ids, "esrb_ratings", esrb)
	else:
		if hasattr(game_rawg, "esrb_rating") and game_rawg.esrb_rating["name"] != "No Rating" and game_rawg.esrb_rating["name"] != "Rating Pending":
			esrb = get_property(names_to_ids, "esrb_ratings", game_rawg.esrb_rating["name"], True)
		elif not hasattr(game_rawg, "esrb_rating"):
			print("No esrb data found from RAWG. Please specify an override using --esrb")
			exit(-1)
			
	if metacritic != -1:
		print("Using user override for metacritic")
		metacritic = float(metacritic)
	else:
		if hasattr(game_rawg, "metacritic"):
			metacritic = float(game_rawg.metacritic)
		else:
			print("No metacritic data found from RAWG. Please specify an override with --metacritic")
			exit(-1)
	
	if release_date != -1:
		print("Using user override for release date")
		release_date = float(release_date)
	else:
		release_date = float(game_rawg.released[0:4])
	
	return genre, esrb, metacritic, release_date

def get_attributes_from_internet(args, names_to_ids):
	series = get_property(names_to_ids, "series", args.series)
	gameplay_main = get_htlb_data(args, names_to_ids)
	genre, esrb, metacritic, release_date = get_rawg_data(args, names_to_ids)
	return series, genre, esrb, gameplay_main, metacritic, release_date

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_model", required=False, help="model to use for inference", default="models/model.pt")
	parser.add_argument("--input_map", required=False, help="name of json file which maps ids to categories, vice versa", default="seeded_data/data.json")
	parser.add_argument("--name", required=False, help="The name to search for in RAWG / Howlongtobeat", default="")
	parser.add_argument("--selector", required=False, help="Which game to pick from RAWG / Howlongtobeat from the results list (0, 1, ...)", default=0)
	parser.add_argument("--series", required=True, help="The game series (Mario Sports, Katamari, etc)")
	parser.add_argument("--genre", required=False, help="The game genre (Racing, RPG, Strategy, etc)", default="None")
	parser.add_argument("--esrb", required=False, help="The ESRB (Everyone, Everyone 10+, Teen, Mature)", default="None")
	parser.add_argument("--gameplay", required=False, help="How long to beat (hours)", default=-1)
	parser.add_argument("--metacritic", required=False, help="The metacritic score (1-100)", default=-1)
	parser.add_argument("--release", required=False, help="The release date (yyyy)", default=-1)
	
	args = parser.parse_args()
	args.selector = int(args.selector)
	args.gameplay = int(args.gameplay)
	args.metacritic = int(args.metacritic)
	args.release = int(args.release)
	
	with open(args.input_map, "r", encoding='utf-8') as f:
		data = json.loads(f.read())["map"]["name-to-id"]
		
	# use data supplied by user
	if args.name == "":
		series, genre, esrb, gameplay_main, metacritic, release_date = get_attributes_from_user(args, data)
		
	# search RAWG / Howlongtobeat for data
	else:
		series, genre, esrb, gameplay_main, metacritic, release_date = get_attributes_from_internet(args, data)
	
	num_series = len(data["series"].keys())
	num_genres = len(data["genres"].keys())
	num_esrb_ratings = len(data["esrb_ratings"])
	sizes = (num_series, num_genres, num_esrb_ratings)
	
	model = TrainingModel(sizes)
	model.load_state_dict(torch.load(args.input_model))
	model.eval()
	
	series_tensor = F.one_hot(torch.tensor([series], dtype=torch.long), len(data["series"].keys())).type(torch.FloatTensor)
	genres_tensor = F.one_hot(torch.tensor([genre], dtype=torch.long), len(data["genres"].keys())).type(torch.FloatTensor)
	esrb_tensor = F.one_hot(torch.tensor([esrb], dtype=torch.long), len(data["esrb_ratings"].keys())).type(torch.FloatTensor)
	continuous = torch.unsqueeze(torch.tensor([gameplay_main,metacritic,release_date], dtype=torch.float32), 0)
	with torch.no_grad():
		out_data = model(((series_tensor, genres_tensor, esrb_tensor), continuous))
		print(out_data)