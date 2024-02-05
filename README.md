# Model-based deep reinforcement learning for accelerated learning from flow simulations

## Introduction
**TODO:**
- documentation

## Data

## Dependencies
MB-version of [drlfoam](https://github.com/JanisGeise/drlfoam/tree/mb_drl)

## Running a training

## Visualize results
- the data is assumed to be located in the directory `data` (tar archive)
- cylinder data in `data/rotatingCylinder2D`, pinball data in `data/rotatingPinball2D`
- the post-processing scripts need to be executed from inside the `post_processing` directory
- the plots are saved under `../plots/rotatingCylinder2D` and `../plots/rotatingPinball2D`, the `plots` directory is created 
automatically when initially executing a post-processing script (path relative to location of the script)
- the scripts are executed for both the cylinder and pinball (no specifications wrt case required), except for visualization
of results from PPO training

the scripts are named as follows:
- `plot_results_cylinder.py`: plots the results of the PPO-training for the 'rotatingCylinder2D' environment 
(rewards, trajectories vs. episodes, ...), plots additionally the results of the final policies compared to MF and uncontrolled case (cl, cd, action vs time , ...)
- `plot_results_pinball.py`: same as `plot_results_cylinder.py`, but for the 'rotatingPinball2D' environment
- `get_mean_cl_cd_final_results.py`: compute the mean and std. deviation of cl and cd for each policy when validated in the
corresponding environment
- `plot_execution_times_of_training.py`: plot a bar chart of the decomposition of the overall execution times wrt MF
- `plot_amount_discarded_trajectories.py`: plot the amount of discarded (invalid) trajectories encountered in each MB training
- `ppo_data_loader.py`: script for loading the results of the PPO training etc., only required as dependency for the
`plot_results_*.py` scripts