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

import rawg.rawgpy as rawgpy
from howlongtobeatpy import HowLongToBeat
import json
import csv
import argparse

def add_to_dict(ids_to_names, names_to_ids, group, name):
	if name not in names_to_ids[group]:
		id = len(names_to_ids[group].keys())
		ids_to_names[group][id] = name
		names_to_ids[group][name] = id
		return id
	return names_to_ids[group][name]

def create_games(input, output):
	with open("rawg_info.json", "r", encoding='utf-8') as f:
		rawg_data = json.loads(f.read())
		user_agent = rawg_data["user-agent"]
		api_key = rawg_data["key"]
	rawg = rawgpy.RAWG(user_agent, api_key)
	
	ids_to_names = {
		"esrb_ratings": {0: "None"},
		"genres": {0: "None"},
		"developers": {0: "None"},
		"publishers": {0: "None"},
		"series": {0: "None"}
	}
	names_to_ids = {
		"esrb_ratings": {"None": 0},
		"genres": {"None": 0},
		"developers": {"None": 0},
		"publishers": {"None": 0},
		"series": {"None": 0}
	}
	games_dict = {}
	
	with open(input, 'r', newline='', encoding='utf-8') as games_file:
		reader = csv.reader(games_file)
		next(reader)
		for row in reader:
			name = row[0]
			series = row[1]
			target_value = int(row[2])
			selector = int(row[3])
			results = rawg.search(name)
			if len(results) == 0:
				print("Could not find entry for \"" + name + "\" on RAWG. Skipping")
				continue
			if len(results) < selector:
				print("a selector of " + selector + " was specified, but there were only " + len(results) + " results. Using first result")
				selector = 0
			game_rawg = results[selector]
			game_rawg.populate()
			results = HowLongToBeat().search(name)
			game_hltb = -1
			if len(results) != 0:
				game_hltb = results[selector]
			else:
				print("Could not find entry for \"" + name + "\" on Howlongtobeat. Entering null value for gameplay time")
			
			game_developers = []
			game_publishers = []
			game_genres = []
			game_esrb = 0
			game_metacritic = -1
			game_gameplay_main = -1
			game_gameplay_completionist = -1
			
			# RAWG does not always have data for these categories
			if hasattr(game_rawg, "metacritic"):
				game_metacritic = game_rawg.metacritic
			if hasattr(game_rawg, "esrb_rating") and game_rawg.esrb_rating["name"] != "No Rating" and game_rawg.esrb_rating["name"] != "Rating Pending":
				game_esrb = add_to_dict(ids_to_names, names_to_ids, "esrb_ratings", game_rawg.esrb_rating["name"])
				
			for genre in game_rawg.genres:
				game_genres.append(add_to_dict(ids_to_names, names_to_ids, "genres", genre.name))
			for developer in game_rawg.developers:
				game_developers.append(add_to_dict(ids_to_names, names_to_ids, "developers", developer.name))
			for publisher in game_rawg.publishers:
				game_publishers.append(add_to_dict(ids_to_names, names_to_ids, "publishers", publisher.name))
			
			game_series = add_to_dict(ids_to_names, names_to_ids, "series", series)
			
			if game_hltb != -1:
				if game_hltb.gameplay_main != -1:
					if game_hltb.gameplay_main_unit != "Hours":
						game_gameplay_main = 1
					else:
						game_gameplay_main = int(game_hltb.gameplay_main.replace("½", ""))
				if game_hltb.gameplay_completionist != -1:
					if game_hltb.gameplay_completionist_unit != "Hours":
						game_gameplay_completionist = 1
					else:
						game_gameplay_completionist = int(game_hltb.gameplay_completionist.replace("½", ""))
			
			key_name = name
			if name in games_dict:
				key_name = name + " " + game_rawg.released[0:4]
			games_dict[key_name] = {
				"name": name,
				"esrb": game_esrb,
				"description": game_rawg.description_raw,
				"release_date": game_rawg.released,
				"metacritic": game_metacritic,
				"genres": game_genres,
				"developers": game_developers,
				"publishers": game_publishers,
				"series": game_series,
				"gameplay_main": game_gameplay_main,
				"gameplay_completionist": game_gameplay_completionist,
				"target_value": target_value
			}
			print("Successfully read \"" + name + "\"")

	final_dict = {
		"games": games_dict,
		"map": {
			"id-to-name": ids_to_names,
			"name-to-id": names_to_ids
		}
	}
	
	print("Finished: writing to disk at " + output)
	with open(output, "w+", encoding='utf-8') as f:
		f.write(json.dumps(final_dict))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=False, help="input .csv file to seed games for", default="raw_data/games.csv")
	parser.add_argument("--output", required=False, help="name (without extension) of the two files to output", default="seeded_data/data.json")
	args = parser.parse_args()
	create_games(args.input, args.output)