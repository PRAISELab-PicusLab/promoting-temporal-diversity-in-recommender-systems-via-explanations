from utils import *

selected_dataset        = "CELLPHONES"          # ML1M | LFM1M | CELLPHONES
selected_recommender    = "NeuMF"         # PGPR | CAFE
# corrective_action       = "MW"            # MW | RR
# corrective_weight       = 0.0             # 0 | 0.125 | 0.25 | 0.375 | 0.5 
user_model              = "LIN"           # UNI | LIN | TOP | EXP | CBM | PBM

final_iteration         = 31
corrective_iteration    = 5
selected_users          = 1000
users_ratings           = 25

for selected_dataset in ["ML1M", "LFM1M", "CELLPHONES"]:
    for corrective_action in ["MW", "RR"]:
        for corrective_weight in [0.0, 0.125, 0.25, 0.375, 0.5]:
            if corrective_weight == 0.0 and corrective_action == "RR":
                continue  # Skip RR when corrective_weight is 0

            combo = f"{selected_dataset} | {selected_recommender} | {corrective_action} | {corrective_weight} | {user_model}"
            print("\n===========================================================================")
            print(combo)
            print("===========================================================================")

            command = [
                sys.executable,
                "Simulation.py",
                "--dataset", str(selected_dataset),
                "--recommender", str(selected_recommender),
                "--corrective_action", str(corrective_action),
                "--corrective_weight", str(corrective_weight),
                "--final_iteration", str(final_iteration),
                "--corrective_iteration", str(corrective_iteration),
                "--num_users", str(selected_users),
                "--num_ratings", str(users_ratings),
                "--user_model", str(user_model),
            ]

            subprocess.run(command, check=True)