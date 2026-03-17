from utils import *

# Available datasets, recommender systems, corrective actions, and weight values
possible_datasets = ["ML1M", "LFM1M", "CELLPHONES"]
possible_recommender = ["PGPR", "CAFE"]
possible_corrective_actions = ['MW', 'RR']
possible_corrective_weights = ['0', '0.125', '0.25', '0.375', '0.5', '0.75', '1']

# Hyperparameters for simulation
corrective_weight = 0.5

final_iteration = 2
corrective_iteration = 2

selected_users = 75
users_ratings = 25

results = []

for selected_dataset in possible_datasets:
    for selected_recommender in possible_recommender:
        for corrective_action in possible_corrective_actions:
            combo = f"{selected_dataset} | {selected_recommender} | {corrective_action} | {corrective_weight}"
            print("\n===========================================================================")
            print(combo)
            print("===========================================================================")

            command = f"python Simulation.py --dataset {selected_dataset} --recommender {selected_recommender} --corrective_action {corrective_action} --corrective_weight {corrective_weight} --final_iteration {final_iteration} --corrective_iteration {corrective_iteration} --num_users {selected_users} --num_ratings {users_ratings}"
            ret = subprocess.run(command, shell=True)
            status = "OK" if ret.returncode == 0 else f"FAILED (exit code {ret.returncode})"
            results.append((combo, status))
            print(f">>> {status}")

print("\n===========================================================================")
print("SUMMARY")
print("===========================================================================")
for combo, status in results:
    print(f"  [{status}]  {combo}")