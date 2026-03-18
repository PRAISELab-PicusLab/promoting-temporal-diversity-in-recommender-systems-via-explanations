from utils import *
import sys

def simulation(dataset, recommender, corrective_action, corrective_weight, final_iteration, corrective_iteration, num_users, num_ratings, user_model):
    prepare_dataset(dataset, recommender, corrective_action, corrective_weight, num_users, num_ratings)

    for iteration in range(1, final_iteration+1):
        start_iteration = time.time()
        print("\n-------------------------------------------------------------------------")
        print(f"ITERATION {iteration}")
        print("-------------------------------------------------------------------------")

        print(f"\nPREPROCESSING")
        print("*************************************************************************")
        command = [sys.executable, "1_Preprocessing.py", "--dataset", str(dataset), "--iteration", str(iteration)]
        subprocess.run(command, check=True)

        print(f"\nRECOMMENDER")
        print("*************************************************************************")
        command = [
            sys.executable, "2_Recommender.py",
            "--dataset", str(dataset),
            "--recommender", str(recommender),
            "--iteration", str(iteration),
            "--corrective_iteration", str(corrective_iteration),
            "--corrective_weight", str(corrective_weight),
        ]
        subprocess.run(command, check=True)

        if (iteration % corrective_iteration == 0 or corrective_action == 'RR') and iteration > 1 and corrective_weight > 0:
            print(f"\nCORRECTIVE ACTION")
            print("*************************************************************************")
            command = [
                sys.executable, "3_Correction.py",
                "--iteration", str(iteration),
                "--corrective_action", str(corrective_action),
                "--corrective_weight", str(corrective_weight),
            ]
            start_time = time.time()
            subprocess.run(command, check=True)
            if corrective_action == 'RR':
                print_elapsed_time(iteration, start_time, 'CORRECTIVE ACTION: RE-RANKING')
            else:
                print_elapsed_time(iteration, start_time, 'CORRECTIVE ACTION: MODIFY WEIGHTS')
        else:
            # Save recommendations without correction
            iteration_folder_path = f"results/analysis/iteration_{iteration}"
            create_directory(iteration_folder_path)
            recommendations_path = f"results/recommendations/iteration_{iteration}.csv"
            if not os.path.exists(recommendations_path):
                raise FileNotFoundError(
                    f"Missing recommendations file: {recommendations_path}. "
                    "The recommender step likely failed in this iteration."
                )
            recommendations_df = pd.read_csv(recommendations_path)
            recommendations_df = recommendations_df.drop(columns=['score', 'paths'])
            recommendations_df = recommendations_df.groupby('uid')['item'].apply(list).reset_index()
            recommendations_df.rename(columns={"item": "items"}, inplace=True)
            recommendations_df.to_csv(f"{iteration_folder_path}/all_recommendation.csv", index=False)

        print(f"\nRECOMMENDATION")
        print("*************************************************************************")
        start_time = time.time()
        command = [sys.executable, "4_Recommendation.py", "--iteration", str(iteration), "--user_model", str(user_model)]
        subprocess.run(command, check=True)
        print_elapsed_time(iteration, start_time, 'SAVING RECOMMENDATIONS')
        print_elapsed_time(iteration, start_iteration, 'ALL PROCESS')

        if iteration == final_iteration:
            copy_simulation_results(dataset, recommender, corrective_action, corrective_weight, user_model)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Selected dataset')
    parser.add_argument('--recommender', type=str, required=True, help='Selected recommender')
    parser.add_argument('--corrective_action', type=str, required=True, help='Selected corrective action')
    parser.add_argument('--corrective_weight', type=float, required=True, help='Selected corrective weight')
    parser.add_argument('--final_iteration', type=int, required=True, help='Final iteration')
    parser.add_argument('--corrective_iteration', type=int, required=True, help='Corrective iteration')
    parser.add_argument('--num_users', type=int, required=True, help='Number of selected users')
    parser.add_argument('--num_ratings', type=int, required=True, help='Minimum number of ratings per user')
    parser.add_argument('--user_model', type=str, default='LIN', help='User choice model')
    
    args = parser.parse_args()
    simulation(args.dataset, args.recommender, args.corrective_action, args.corrective_weight,
               args.final_iteration, args.corrective_iteration, args.num_users, args.num_ratings, args.user_model)

if __name__ == "__main__":
    main()
