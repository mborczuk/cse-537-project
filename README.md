# Tetris using Classification Models

## Original Code Base

The original code was obtained from this project: https://github.com/rovle/models-playing-tetris/tree/main.

We mostly only used this repository for the Tetris interface, which we modified in order to log data as well as collect data on model performance.

## Modifications

We only modified the tetris/game.py file, specifically the act method in the Game class. This is the method that is called when a new piece spawns and when the user moves/rotates a piece. We changed this function to log the piece, board state, and the move the user made - these logs are what were parsed to create our dataset. In addition, we changed this function to add a call to the predict function, which queries our trained model(s) for what move to make next based on the current board state and piece.

## Training and Testing the Models

All models were trained and tested using the scikit-learn package.

To train our models, we simply edit line 15 in the ```tetris/game.py``` file to read from either ```prediction``` (base classification models), ```prediction_mul_random``` (trenchcoat with random forest), or ```prediction_mul``` (trenchcoat with optimized classifiers). So, line 15 should read ```from prediction import predict```, ```from prediction_mul_random import predict```, or ```from prediction_mul import predict```. In addition, by editing prediction.py, prediction_mul_random.py, or prediction_mul.py, the dataset to use can be selected. The three available datasets are located in ```logs/samples.json``` (small dataset we used for testing), ```logs/samples-julia.json``` (dataset we used for base classification models), and ```logs/samples-julia-outline.json``` (dataset we used for outline only and trenchcoat models).

Once these settings are set, the code can be run with ```python main.py``. This will launch a new tetris game, train the selected model on the selected dataset, and then start outputting the moves it thinks the user should make.

## Models and Datasets Available
### Models
Random - can be run with ```python main.py mode=random```; from the original paper 
LLM (Google Gemini) - can be run with ```python main.py --model gemini-1.5-flash --temperature 0.4 --prompt_name complex_cot_prompt_n5_multiple_actions_v1```; from the original paper 
Random Forest - can be run by editing line 33 of ```prediction.py``` to use ```RandomForestClassifier()```; use ```samples-julia.json``` on line 11; use ```from prediction import predict``` on line 15 of ```tetris/game.py``` 
K Neighbors - can be run by editing line 33 of ```prediction.py``` to use ```KNeighborsClassifier(n_neighbors=37)```; use ```samples-julia.json``` on line 11; use ```from prediction import predict``` on line 15 of ```tetris/game.py``` 
Multi-Layer Perceptron - can be run by editing line 33 of ```prediction.py``` to use ```MLPClassifier(hidden_layer_sizes=(100, 100))```; use ```samples-julia.json``` on line 11use ```from prediction import predict``` on line 15 of ```tetris/game.py``` 
Outline Only - can be run by editing line 33 of ```prediction.py``` to use ```RandomForestClassifier()```; use ```samples-julia-outline.json``` on line 11use ```from prediction import predict``` on line 15 of ```tetris/game.py``` 
Trenchcoat (Random Forest) - use ```from prediction_mul_random import predict``` on line 15 of ```tetris/game.py``` 
Trenchcoat (Optimized) - use ```from prediction_mul import predict``` on line 15 of ```tetris/game.py``` 

### Datasets
```samples.json``` - test dataset, do not use 
```samples-julia.json``` - dataset used to test first 3 classifiers; encodes the entire board matrix 
```samples-julia-outline.json``` - dataset used to test last 3 classifiers; encodes only the heights of each board column 

