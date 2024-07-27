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


Environment Rules (Simulation - How the solution's payoff matrix affected)
-----------------------------------------------------------------------------
1. Blocks with many neighbors are more difficult (time-consuming) to color than blocks with fewer neighbors.
   Over the stages of the game, ideally, the human should have collected almost the same amount of difficulty 
   points as the robot. During the simulation, each player collects their difficulty points, and at the end of
   the game, the difference between those points signifies how much extra time the game needed to end. "Extra 
   time" means that at some point, one player had finished coloring while the other continued for that extra 
   time. If this difference is very small, it might be the bare minimum in some environments where a difference 
   of exactly 0 is impossible. However, this is not usually the case. So when dealing with hundreds of simulations, 
   the mean extra value represents well how time-consuming a pair of players is.
   
   [payoff = payoff - extra time]

2. Blocks with even IDs should ideally be colored with warm colors (colors with a dominant red component), e.g., 
   Red, Orange, Yellow, Coral, Pink, whereas blocks with odd IDs should ideally be colored with cool colors (colors 
   with a dominant blue or green component), e.g., Green, Blue, Violet, Cyan, Teal. In the final solution of the 
   game, we count a penalty k for every wrong ID-color assignment found.
   
   [payoff = payoff - k * number of wrong id-color assignments]

3. Blocks should be colored with the minimum number of distinct colors for efficiency. In the final solution, we 
   count a penalty k if k different colors are present on the grid.

Agents' Response (Learning - How the players' reward function is affected)
---------------------------------------------------------------------------------

1. Agents have preferences for blocks based on their difficulty. For example, an agent might prefer easier blocks 
   over harder ones and therefore reward themselves more for choosing to color an easier block. For a pair to work 
   ideally well together, we expect their preferences to complement each other.

2. Agents have preferences for colors based on their own taste. For example, an agent might prefer Red over Blue 
   and therefore reward themselves more for choosing Red. For a pair to work ideally well together, we expect both 
   players to have preferences that are compatible with the game rules regarding the even-warm and odd-cool block 
   color assignment logic.

3. Agents have preferences for either minimalism or diversity when choosing colors. For example, an agent might 
   reward themselves with +k or -k for choosing a color that is already present k times in the grid.

Mapping to Air Traffic Management
---------------------------------

(a) Blocks / Airspaces

   Blocks in the grid represent distinct sections of airspace. Each airspace is a defined volume where aircraft 
   are managed and monitored. Neighboring blocks can be understood as adjacent airspaces, where managing one 
   airspace might impact the management of neighboring airspaces due to potential conflicts and coordination 
   requirements.

(b) Agents / Controllers

   Agents in the game are similar to air traffic controllers. Assigning color to a block represents the way
   controllers resolve conflicts in air traffic by assigning different altitudes, routes, or speeds to aircraft.

(c) Colors / Strategy

   Colors represent different strategies used to resolve conflicts. These strategies are categorized into two 
   main classes: warm colors and cool colors. Each class represents a different set of strategies with common 
   characteristics. In the context of the game, colors are only an abstraction for behaviors and not the actual 
   behaviors, so they shall not be further defined.

(d) Rules explained

   #1: The complexity and workload of managing a particular airspace depend on the number of neighboring airspaces. 
   More neighboring airspaces increase the potential for conflicts and require more coordination. To ensure 
   efficiency and safety, it is crucial to distribute the workload evenly among controllers.

   #2: Certain airspaces (with even or odd IDs) might be better suited to specific conflict resolution strategies 
   (warm or cool colors). This logic could represent established protocols where specific types of airspaces,
   such as high-traffic and low-traffic, are managed using suitable strategies that optimize efficiency and safety.

   #3: When controllers use a limited set of strategies, there's a consistent approach to managing air traffic.
   consistency makes it easier for controllers to predict and understand each other’s actions, reducing 
   misunderstandings and errors.

"""