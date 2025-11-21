import sensor_sim
import numpy as np
import pandas as pd
import os

# config
WINDOW_SIZE = 64
SAMPLE_RATE = 100.0
SAMPLES_PER_CLASS = 1000

# Activities we want to learn
# Format: (Name, Label_ID)
ACTIVITIES = [
    ("sitting", 0),
    ("walking", 1),
    # You can add "running" or "jumping" if you updated the C++ logic!
]

def generate_data():
    # init C++ modules
    sim = sensor_sim.IMUSim(SAMPLE_RATE, "sitting")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SAMPLE_RATE)
    dataset = []
    print(f"--- Generating {SAMPLES_PER_CLASS} samples per class ---")
    for act_name, label_id in ACTIVITIES:
        print(f"Generating: {act_name}...")
        sim.set_curr_activity(act_name)
        # fill buffer initially
        for _ in range(WINDOW_SIZE):
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())
        # now collect samples
        for i in range(SAMPLES_PER_CLASS):
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())
            # grab curr window's features
            feats = extractor.compute_features()
            # Save: [Features..., Label]
            row = list(feats) + [label_id]
            dataset.append(row)
    # convert to df, save
    cols = ["Mean_Az", "Var_Az", "Dom_Freq", "Energy", "Label"]
    df = pd.DataFrame(dataset, columns=cols)
    output_file = "imu_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset saved to {output_file}")
    print(df.head())
    print("\nClass distribution:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    generate_data()