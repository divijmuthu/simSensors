import sensor_sim
import numpy as np
import pandas as pd
import os

# config
WINDOW_SIZE = 256
SAMPLE_RATE = 100.0
SAMPLES_PER_CLASS = 1000

# Activities: (Name, Label_ID)
ACTIVITIES = [
    ("sitting", 0),
    ("walking", 1),
    ("running", 2), 
    ("jumping", 3),
    ("stairs_up", 4),
    ("elevator_up", 5)
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
        
        # Train on MULTIPLE directions so the model isn't biased to East
        directions = [1.0] # Default
        if act_name in ["walking", "running"]:
            directions = [1.0, -1.0] # East and West
            
        samples_per_dir = SAMPLES_PER_CLASS // len(directions)
        
        for direction in directions:
            # Set velocity based on activity AND direction
            if act_name == "walking":
                sim.set_velocity(1.5 * direction)
            elif act_name == "running":
                sim.set_velocity(3.0 * direction)
            else:
                sim.set_velocity(0.0)
            
            # Warm up buffer
            for _ in range(WINDOW_SIZE):
                sim.update()
                extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope(), 
                                     sim.get_magnetometer(), sim.get_pressure())
                
            # Collect samples
            for i in range(samples_per_dir):
                sim.update()
                extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope(), 
                                     sim.get_magnetometer(), sim.get_pressure())
                
                feats = extractor.compute_features()
                row = list(feats) + [label_id]
                dataset.append(row)
            
    # Save to CSV
    cols = ["Mean_Az", "Var_Az", "Dom_Freq", "Energy", "Vert_Vel", "Mean_MagX", "Mean_MagY", "Label"]
    
    df = pd.DataFrame(dataset, columns=cols)
    output_file = "imu_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Dataset saved to {output_file}")
    print(df.head())
    print("\nClass distribution:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    generate_data()