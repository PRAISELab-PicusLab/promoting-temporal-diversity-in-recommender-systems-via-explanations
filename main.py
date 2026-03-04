from utils import *

# Available datasets, recommender systems, corrective actions, and weight values
possible_datasets = ["ML1M", "LFM1M", "CELLPHONES"]
possible_recommender = ["PGPR", "CAFE"]
possible_corrective_actions = ['MW', 'RR']
possible_corrective_weights = ['0', '0.125', '0.25', '0.375', '0.5', '0.75', '1']

# Hyperparameters for simulation
selected_dataset = "ML1M"
selected_recommender = "CAFE"
corrective_action = "RR"
corrective_weight = 0.5

final_iteration = 1
corrective_iteration = 2
trained_model = 0

selected_users = 75
users_ratings = 25

print("\n===========================================================================")
print(f"{selected_dataset.upper()} | {selected_recommender.upper()} | {corrective_action.upper()} | {corrective_weight}")
print("===========================================================================")

command = f"python Simulation.py --dataset {selected_dataset} --recommender {selected_recommender} --corrective_action {corrective_action} --corrective_weight {corrective_weight} --final_iteration {final_iteration} --corrective_iteration {corrective_iteration} --initial_model {trained_model} --num_users {selected_users} --num_ratings {users_ratings}"
subprocess.run(command, shell=True)