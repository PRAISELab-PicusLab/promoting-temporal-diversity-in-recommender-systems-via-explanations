from utils import *

selected_dataset        = "ML1M"          # ML1M | LFM1M | CELLPHONES
selected_recommender    = "NeuMF"         # PGPR | CAFE
corrective_action       = "MW"            # MW | RR
corrective_weight       = 0.5             # 0 | 0.125 | 0.25 | 0.375 | 0.5 | 0.75 | 1
user_model              = "LIN"           # UNI | LIN | TOP | EXP | CBM | PBM

final_iteration         = 3
corrective_iteration    = 2
selected_users          = 750
users_ratings           = 25

# for user_model in ["UNI", "LIN", "TOP", "EXP", "CBM", "PBM"]:
combo = f"{selected_dataset} | {selected_recommender} | {corrective_action} | {corrective_weight} | {user_model}"
print("\n===========================================================================")
print(combo)
print("===========================================================================")

command = (
    f"python Simulation.py"
    f" --dataset {selected_dataset}"
    f" --recommender {selected_recommender}"
    f" --corrective_action {corrective_action}"
    f" --corrective_weight {corrective_weight}"
    f" --final_iteration {final_iteration}"
    f" --corrective_iteration {corrective_iteration}"
    f" --num_users {selected_users}"
    f" --num_ratings {users_ratings}"
    f" --user_model {user_model}"
)

subprocess.run(command, shell=True)