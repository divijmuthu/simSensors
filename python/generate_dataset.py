import sensor_sim
import numpy as np
import pandas as pd
import os

# config
WINDOW_SIZE = 256
SAMPLE_RATE = 100.0
# high sample count for robustness
SAMPLES_PER_CLASS = 5000 

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
        
        # velocity must be incorporated effectively to distinguish activities 
        # e.g. standing vs walking & running, stairs vs elevator
        if act_name == "walking":
            directions = [1.5, -1.5] # Normal Speed
        elif act_name == "running":
            directions = [3.0, -3.0] # Fast Speed
        elif act_name == "stairs_up":
            directions = [0.5, -0.5] # Slow Walk for Stairs
        else:
            directions = [0.0] # Sitting, Jumping, Elevator are stationary
            
        samples_per_dir = SAMPLES_PER_CLASS // len(directions)
        
        for velocity in directions:
            sim.set_velocity(velocity)
            
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