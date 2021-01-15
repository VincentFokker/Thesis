# Thesis Vanderlande 
This repository contains all the code necessary to reproduce the RL agent and the simulation used during training.

To run, setup a virtual environment with:
- Python 3.6.8
- packages available in **Requirements.txt**

(install with: `$pip install -r requirements.txt` )
## Folders in repository
-  **evaluation_results** \
Folder with the evaluations of  the DRL agent and the Heuristic Agent on the Idle time and Recirculation of carriers.
- **figures** \
All the generated figures also used in the report.
- **Notebooks** \
Jupyter notebooks used for generating intermediate results, for testing the Heuristic Agent and for thinkering with the environment.
- **Renders** \
Folder with generated renders of the simulation for visual inspection.
- **rl** \
All the Reinforcement Learning script, including the environments, the configuration, helper scripts, results of the hyper_parameter optimization with Baysian Optimization and a Folder with the relevant trained models.


##  Explaination of scripts in repository

- **check_env.py** \
 A script to check if an environment in `rl/environements` adheres to the format of the OpenAI Gym framework. Run this script with: \
 `python check_env.py -e [ENVIRONMENT]` 
 
- **check_performance.py** \
Runs a test on a given environment, has possibility to render. Usage is: \
`python check_performance.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [RUNNR]`

- **expert_trajectories.py** \
Script to generate expert trajectories on an given environment. The Heuristic agent is taken as the expert to generate trajectories. The number of episodes to generate can be given as an input parameter. Run as:
`python expert_trajectories.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -c [CONFIG] -n [NUMEPISODES]`

- **hyperparameter.py** \
Script that uses the hyperdrive package to do bayesian optimization on the weights of the reward shaping function. Can be run by: \
`python hyperparameter.py -e [ENVIRONMENT_NAME]`

- **plotcombiner_x.py** \
Script to generate and combine plots for the rewards and the loss subtracted from the tensorboard logs. Figures are stored in `/figures/`.

- **plotmaker.py** \
Script to make a reward and loss plot for a specific result in the trained_models folder. Usage of this script: \
`python plotmaker.py -e [ENVIRONMENT_NAME] -s [SUBDIR]`

- **pretrain.py** \
Script that is used to do the pre-training of an PPO agent with behavioral cloning, based on the data generated with the `expert_trajectories.py` script. Retraining after the pretraining phase is also possible with the `-r` argument.
Run with: \
`python pretrain.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD] -r [RETRAIN]`

- **resultmaker.py** \
Script to generate the results on the objectives performance for the trained DRL agents. Runs 100 episodes and averages the results over these 100 episodes to obtain mean performance. Results are stored in `/evaluation_results/`. Can be run by: \
`python resultmaker_x.py -t [TERM] -n [NUM_EPISODES]`

- **retrain_callback.py** \
Function to retrain a model in a specific subfolder in the `/trained_models/` directory.  Has an evaluation callback during training to evalute the model every 10 000 steps. Can be run by: \
`python retrain_callback.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD]`

- **test_all.py** \
Script that reads all the models from a subdir in `trained_models` and tests the model with possibility to render the output. The action can be done deterministic or stochastic, depending of parameter `-d`. Can be run by: \
`python test_all.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -r -d`

- **train_callback.py** \
Script to train an agent on a given environment. Uses a callback to store the best model, the model is evaluated each 10 000 steps. The model is also saved in a pre-defined interval. Can be run with: \
`python train_callback.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD]`

- **train_on.py** \
Script to train the agent, with two callbacks; one for saving the best model and one for terminating the training when the evaluation result is over a threshold of 95% of the maximum reward that can be obtained. Run by: \
`python train_on.py -e [ENVIRONMENT_NAME] -s [SUBDIR] -n [LOGNAME] -c [CONFIG] -t [TENSORBOARD]`


 