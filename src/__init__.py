"""
Project Description

This project is designed to provide a simple and user-friendly interface for doing research on the cgc-game.

User's Perspective
-------------------

The user should be able to:

1. Edit a simple configuration file (`config.yaml`) to change game parameters (environment, players, etc.).
2. Run the following commands:
   - `poetry run learner`
   - `poetry run simulator`
   - `poetry run evaluator`

The user should not need to access any other files or functionalities. All details related to how the game works 
internally during training, simulation, and evaluation are hidden from the user.

The order in which the processes occur should be strictly as follows: learning > simulation > evaluation. The 
`simulator.py` and `evaluator.py` scripts should include checks to ensure that the required output files from 
the learning and simulation processes, respectively, exist. If the files are missing, an error message should
be displayed to the user, and the process should not proceed.

Project Structure
-----------------

The repository is organized as follows:

cgcg/
├── src/
│   ├── __init__.py        
│   ├── learner.py     
│   ├── simulator.py   
│   ├── evaluator.py   
│   ├── utils.py       
│   ├── grid.py         
│   ├── player.py      
│   ├── game.py        
│   ├── colors.py      
│   ├── model.py      
├── config.yaml         
├── README.md            
├── pyproject.toml        

"""