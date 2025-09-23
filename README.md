# 🌐 Promoting Temporal Diversity in Recommender Systems via Explanations  

The goal of the project is to maximize long-term diversity in recommendations by monitoring and manipulating how explanations evolve over time.

## ✨ Project Objectives 

Knowledge-graph-based recommendation systems produce accurate suggestions but tend to repeat similar content over time. Our approach combines explainable models (like PGPR and CAFE) with analysis of explanation variety to:

- Measure each item’s impact on future diversity. A Segregation Score is computed to evaluate how much an item leads the user to a narrow area of the graph.
- Detect repetitive explanation paths. We analyse the model’s reasoning paths and measure the overlap between successive iterations.
- Apply corrective actions. When diversity falls below a threshold, we intervene by reordering the recommendations (RR – Re‑ranking) or modifying the weights of certain relations in the graph (MW – Adjusting item-related weights).
- Evaluate dynamics over time. The pipeline runs multiple iterations (simulating successive user interactions) and records the evolution of accuracy and diversity.

## 🗂 Repository Structure

- **main.py**  
  Defines high-level parameters (model, dataset, number of users, number of iterations, type and strength of corrective action) and starts the full pipeline. Creates the necessary directory structure, filters the users, and launches preprocessing, training, and correction scripts.

- **1_Preprocessing.py**  
  Loads the filtered `users.csv`, `items.csv`, and `ratings.csv` files and remaps the IDs to a dense index space. Produces `users.txt`, `products.txt`, and `ratings.txt` for later stages.

- **2_Models.py**  
  Handles the time-based train/test split, prepares the data for PGPR or CAFE models, trains (or reuses) TransE embeddings and the PGPR agent or the CAFE neural model. Extracts explanation paths and aggregates them into recommendations. Accepts parameters such as model, dataset, and retraining frequency.

- **3_Correction.py**  
  Implements the two corrective actions:  
  - **RR (Re‑ranking):** penalizes items whose explanations change little between iterations, subtracting a score and reordering the list.  
  - **MW (Adjusting item-related weights):** updates the item score matrix by penalizing common paths in the explanations and regenerates the recommendations.

- **4_Recommendation.py**  
  Filters the recommendations by excluding items already in the test set. Simulates the user’s choice according to a decreasing distribution over positions and updates the dataset with the new rating.

- **utils.py**  
  Contains support functions for directory creation, timing, probability distributions, etc.

- **mapper.py**  
  Defines the list of entities and relations for each dataset and prepares the data for explainable models.

- **requirements.txt**  
  Complete list of Python libraries required to run the code.

- **dataset/**  
  Directory to place the original datasets. The pipeline currently supports three sets:  
  - **ML1M** (MovieLens 1M)  
  - **LFM1M** (LastFM 1M)  
  - **CELLPHONES** (extracted from Amazon)  
  
  Each folder must contain raw files (e.g., `users.dat`, `movies.dat`, `ratings.dat` for ML1M).

- **results/**  
  Will contain the recommendations, analyses, and timing for each iteration.

- **process/**  
  Temporary directory hosting:  
  - `csv/` → filtered data  
  - `preprocessed/` → preprocessed datasets  
  - `train_test_set/` → train/test split  
  - `trained_model/` → trained TransE models

## 🛠 Installation

1. **Clone the repository**  

   ```bash
   git clone https://github.com/diegorusso95/promoting-temporal-diversity-in-recommender-systems-via-explanations.git
   cd promoting-temporal-diversity-in-recommender-systems-via-explanations

2. **Prepare the environment**

   ```bash
    pip install -r requirements.txt

## 🚀 Running the Pipeline

The full pipeline is started via `main.py`.  
The most important parameters are:

- **`--model`** → `"PGPR"` or `"CAFE"`
- **`--dataset`** → one of `"ML1M"`, `"LFM1M"`, `"CELLPHONES"`
- **`--corrective_action`** → `"RR"` (Re-Ranking) or `"MW"` (Modify Weights)
- **`--corrective_weight`** → value between `0` and `1` that balances accuracy and diversity
- **`--users` / `--ratings`** → number of users to analyse and minimum rating threshold to be included
- **`--iteration`**, **`--final_iteration`**, **`--corrective_iteration`** → define how many iterations to run and when to apply the correction
