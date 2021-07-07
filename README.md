# DeepGameInference

Want to predict whether you will like a particular game or not (without whoring out your data to Steam)? Try DeepGameInference!

This project can be used to query game data from the **RAWG** database (https://rawg.io/) and **Howlongtobeat** (https://howlongtobeat.com/), and then train a neural network on the data to predict a linear target value [0,10].

The intended use case of this project is to create models to predict how much a person will like a particular game, based on previous games they've played and how much they liked them. For your own projects, the target value can mean anything you like!

## License

This project uses the **GPL Version 3** License.

## Installation

This is a **Python3** project. Install dependencies using

```
pip install -r requirements.txt
```

Initialize the `rawg-python-wrapper` submodule with
```
git submodule update --init
```

## Example files

Example files are provided for each stage of the project, allowing you to skip seeding or training if you are only interested in understanding how the project works. If you want to test out the end result before starting the project, use

```
python inference.py --input_model "models/example_model.pt" --input_map "seeded_data/example_data.json" --series "Final Fantasy" --genre "RPG" --esrb "Teen" --gameplay 30 --metacritic 80 --release 2030
```

The output should be `tensor(8.5483)`, which means that the model predicts a target value of 8.5483.

## Getting an API Key

To create the training data which the neural network is trained on, you will need an API key from RAWG. Visit (https://rawg.io/apidocs) for instructions. You will have to make an account with them and supply an email and password. Once you have obtained your API key, **create a JSON file called `rawg_info.json`** in the base directory with two key-value pairs: `"key"` and `"user-agent"`. The `key` property should associate with your API key from RAWG, and the `user-agent` property should associate with a unique name that describes your "app" to RAWG, so to speak. The string will be included in GET requests to RAWG as header information.

```
{
	"key": "put your api key here",
	"user-agent": "put your user agent here"
}
```

## Seeding

Once you have obtained an API key from RAWG, you will need to make a file containing the names of all the games you would like to be part of your dataset. Each game must have an associated game series (str), target value [0-10], and a "selector" [0-inf). It should be formatted as a .csv, and there is an example in `raw_data/example_games.csv`. It looks like so:

```
Name,Series,Target Value,Selector
Super Smash Bros Melee,Super Smash Bros.,10,0
Doom,Doom,7,1
Doom,Doom,10,0
Stanley Parable,None,8,0
Among Us,None,0,0
```

The first value in the .csv is skipped, so be sure to include the header in your own as well. Note that you are required to specify a series for each game because RAWG doesn't maintain good series data for their games unfortunately; but this gives you the freedom of using the series parameter as any arbitrary grouping mechanism you want. Some ideas include platform (Nintendo, Sony, Xbox, PC, Mobile, etc) or country of origin (America, Japan, etc)

Additionally, note that game names are used to search in RAWG and Howlongtobeat, and the "selector" is used to index into the list of results. You will want this to be 0 for most cases, but this is needed for differentiating between remakes (as seen above, where the user wants the original Doom and the Doom remake to both be used). Typically, game remakes/modernizations will be listed as the first result, and the original game will be listed as the second result, so use "1" for the selector to specify that you are referring to the original game, and use "0" for the selector to specify that you are referring to the remake.

Once you have created your .csv file, query the RAWG/Howlongtobeat data using

```
python seed.py --input "path/to/filename.csv" --output "path/to/filename.json"
```

which will output a file named `filename.json` in the `path/to/` directory. The default input and output is `raw_data/games.csv` and `seeded_data/data.json`.

## Training

With a list of seeded games and a categories map stored in a .json file, you can train the neural network to produce a model. This can be done with

```
python train.py --input "path/to/filename.json" --output "path/to/filename.pt"
```

The default input and output is `seeded_data/data.json` and `models/model.pt`

## Inference

With a categories map stored in a .json file and a trained model, you can infer the target value of new games. This command will query data from RAWG and Howlongtobeat for a new game, transform its properties into indices via the map, and then use the model to infer its target value.

```
python inference.py --input_model "path/to/somewhere/filename.pt" --input_map "path/to/somwhere/filename.json" --name "Luigi's Mansion 3" --series "Luigi's Mansion"
```

Alternatively, you can specify properties directly without querying RAWG or Howlongtobeat. This can be used to do inference for games which do not exist.

```
python inference.py --input_model "path/to/somewhere/filename.pt" --input_map "path/to/somwhere/filename.json" --series "Mario Kart" --genre "Racing" --esrb "Everyone" --gameplay 10 --metacritic 90 --release 2030
```

The default inputs for model and map are `models/model.pt` and `seeded_data/data.json`.

## Ideas for modifications

Currently the neural network does not take a game's developers or publishers into account; it also does not look at a game's name, or the description of the game. Many of these properties are likely redundant to the "series" property, but it would be interesting to see if they improve a model's accuracy or just slow down training. The description property is likely the best candidate for improving accuracy, but would be the most time consuming to implement.

Future modifications I plan to make include adding support for games with multiple genres (training currently ignores all except the first genre), as well as introducing non-linear modules into the TrainingModel.
