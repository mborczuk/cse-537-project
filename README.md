# Tetris using Classification Models

## Original Code Base

The original code was obtained from this project: https://github.com/rovle/models-playing-tetris/tree/main.

We mostly only used this repository for the Tetris interface, which we modified in order to log data as well as collect data on model performance.

## Modifications

We only modified the tetris/game.py file, specifically the ```act``` method in the ```Game``` class. This is the method that is called when a new piece spawns and when the user moves/rotates a piece. We changed this function to log the piece, board state, and the move the user made - these logs are what were parsed to create our dataset. In addition, we changed this function to add a call to the predict function, which queries our trained model(s) for what move to make next based on the current board state and piece.

## Setting Up the Environment
```
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# install the dependencies
pip install -r requirements.txt

# set your API keys in .env (not necessary for classifiers, only necessary for running the original paper's models)
cp .env.example .env
```
## Generating Log Data
When playing through the interface, the game will automatically log all moves to ```logs/log.txt```. The ```logs/read-data.py``` file can convert the raw log data into a regular dataset (JSON) usable by the models, and the ```logs/read-data-outline.py``` file can convert the raw log data into an outline dataset (JSON) usable by the models.

## Training and Testing the Models

All models were trained and tested using the scikit-learn package.

To train our models, we simply edit line 15 in the ```tetris/game.py``` file to read from either ```prediction``` (base classification models; use the ```classifier_mode``` variable on line 17 of ```tetris/game.py``` to switch between them), ```prediction_mul_random``` (trenchcoat with random forest), or ```prediction_mul``` (trenchcoat with optimized classifiers). So, line 15 should read ```from prediction import predict```, ```from prediction_mul_random import predict```, or ```from prediction_mul import predict```. In addition, by setting the ```dataset_mode``` variable on line 18 ```tetris/game.py```, the dataset to use can be selected (this only works for the base classification models, however). There are two main datasets to choose from, ```logs/samples-julia.json``` (regular dataset; set ```dataset_mode = 0```) and ```logs/samples-julia-outline.json``` (outline dataset; set ```dataset_mode = 1```). There is also a validation set located in ```logs/samples.json```, but the code is not currently set up to read from it.

Once these settings are set, the code can be run with ```python main.py --mode=manual```. This will launch a new tetris game, train the selected model on the selected dataset, and then start outputting the moves it thinks the user should make.  

The move output is in the form of ```x:y```, where ```x``` is the number of spaces to move (a negative number means left, a positive number means right) and ```y``` is the number of clockwise rotations to make.

## Models and Datasets Available
### Models
Base classification models can be toggled using the ```classifier_mode``` variable on line 17 of ```tetris/game.py```. The dataset used can be toggled using the ```dataset_mode``` variable on line 18 of ```tetris/game.py```. Please note that these mode variables only have an effect if using the base classification model file (```prediction.py```).
- Random 
    - ```python main.py --mode=random```
    - from the original paper  
- LLM (Google Gemini)
    - ```python main.py --model gemini-1.5-flash --temperature 0.4 --prompt_name complex_cot_prompt_n5_multiple_actions_v1```
    - from the original paper  
- Random Forest
    - Use ```from prediction import predict``` on line 15 of ```tetris/game.py```
    - Set ```classifier_mode``` to 0 
    - Set ```dataset_mode``` to 0 
- K Neighbors
    - Use ```from prediction import predict``` on line 15 of ```tetris/game.py```
    - Set ```classifier_mode``` to 1 
    - Set ```dataset_mode``` to 0  
- Multi-Layer Perceptron
    - Use ```from prediction import predict``` on line 15 of ```tetris/game.py```
    - Set ```classifier_mode``` to 2 
    - Set ```dataset_mode``` to 0   
- Outline Only
    - Use ```from prediction import predict``` on line 15 of ```tetris/game.py```
    - Set ```classifier_mode``` to 0 
    - Set ```dataset_mode``` to 1 
- Trenchcoat (Random Forest) 
    - Use ```from prediction_mul_random import predict``` on line 15 of ```tetris/game.py```  
- Trenchcoat (Optimized) 
    - Use ```from prediction_mul import predict``` on line 15 of ```tetris/game.py```  

### Datasets
- ```samples.json``` 
    - Validation dataset, do not use  
- ```samples-julia.json``` 
    - Dataset used to test first 3 classifiers
    - Encodes the entire board matrix  
- ```samples-julia-outline.json```
    - Dataset used to test last 3 classifiers
    - Encodes only the heights of each board column  

## Requirements
All requirements can be found in requirements.txt.

## Other Files
- ```logs/julia.txt```
    - Raw log data that was converted into ```logs/samples-julia.json``` and ```logs/samples-julia-outline.json```
- ```model.py```
    - Initial base classifier model we wrote, later adapted into ```prediction.py```
- ```model_mul.py```
    - Initial trenchcoat model we wrote, later adapted into ```prediction_mul_random.py``` and ```prediction_mul.py```