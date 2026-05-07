# ===============================
#           MAIN FILE
# ===============================

import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
# OPTIONS(featurez)
feature_dict = {
    1: ['Annual_Spending', 'Purchase_Frequency'],
    2: ['Annual_Spending', 'Returns_Count'],
    3: ['Purchase_Frequency', 'Discount_Usage_Pct'],
    4: ['Age', 'Annual_Spending'],
    5: ['Age', 'Purchase_Frequency']
}

# LOADing MODELs
def load_kmeans(features):
    key = "_".join(features)
    model = pickle.load(open(f"{key}_kmeans_model.pkl", "rb"))
    scaler = pickle.load(open(f"{key}_kmeans_scaler.pkl", "rb"))
    labels = pickle.load(open(f"{key}_kmeans_labels.pkl", "rb"))
    return model, scaler, labels


def load_dbscan(features):
    key = "_".join(features)
    model = pickle.load(open(f"{key}_dbscan_model.pkl", "rb"))
    scaler = pickle.load(open(f"{key}_dbscan_scaler.pkl", "rb"))
    labels = pickle.load(open(f"{key}_dbscan_labels.pkl", "rb"))
    return model, scaler, labels


# INPUTs
def get_feature_input(features):
    values = []
    for f in features:
        try:
            val = float(input(f"Enter {f}: "))
            values.append(val)
        except:
            print("Invalid input! Try again.")
            return None
    return np.array([values])


# KMEANS
def predict_kmeans(features):
    model, scaler, labels = load_kmeans(features)

    data = get_feature_input(features)
    if data is None:
        return

    data_scaled = scaler.transform(data)
    cluster = model.predict(data_scaled)[0]

    print(f"\nCustomer belongs to: {labels[cluster]}")


# DBSCAN
def predict_dbscan(features):
    model, scaler, labels = load_dbscan(features)

    data = get_feature_input(features)
    if data is None:
        return

    data_scaled = scaler.transform(data)

    # DBSCAN (assigning nearest cluster 
    distances = []
    for i, core_point in enumerate(model.components_):
        dist = np.linalg.norm(data_scaled - core_point)
        distances.append(dist)

    if len(distances) == 0:
        print("\nNo clusters found by DBSCAN.")
        return

    nearest_index = np.argmin(distances)
    cluster = model.labels_[nearest_index]

    if cluster == -1:
        print("\nCustomer classified as: Noise / Outlier 🚫")
    else:
        label = labels.get(cluster, "Unknown Segment")
        print(f"\nCustomer belongs to: {label}")


# --------- MAIN MENU ---------
def menu():
    while True:
        print("\n===============================")
        print(" CUSTOMER SEGMENTATION SYSTEM")
        print("===============================")
        print("1. Use KMeans")
        print("2. Use DBSCAN")
        print("3. Exit")

        algo_choice = input("Select algorithm: ")

        if algo_choice == '3':
            print("Exiting... ")
            break

        if algo_choice not in ['1', '2']:
            print("Invalid choice!")
            continue

        print("\nSelect feature combination:")

        for k, v in feature_dict.items():
            print(f"{k}. {v[0]} & {v[1]}")

        try:
            f_choice = int(input("Enter choice: "))
        except:
            print(" Invalid input!")
            continue

        if f_choice not in feature_dict:
            print(" Invalid feature choice!")
            continue

        features = feature_dict[f_choice]

        if algo_choice == '1':
            predict_kmeans(features)
        else:
            predict_dbscan(features)


# --------- RUN ---------
menu()