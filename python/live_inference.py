import torch
import torch.nn as nn
import numpy as np
import sensor_sim
import time

# prep the model architecture 
class ActivityClassifier(nn.Module):
    def __init__(self):
        super(ActivityClassifier, self).__init__()
        self.layer1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def run_live_demo():
    # load in model, eval mode
    print("--- Loading Model... ---")
    checkpoint = torch.load("activity_model.pth", weights_only=False)
    model = ActivityClassifier()
    model.load_state_dict(checkpoint['model_state'])
    model.eval() 
    
    # Scale live data to match training
    train_mean = checkpoint['mean']
    train_std = checkpoint['std']
    
    labels = {0: "SITTING", 1: "WALKING"}
    
    # set up C++ module + feature extractor
    sim = sensor_sim.IMUSim(100.0, "sitting")
    extractor = sensor_sim.FeatureExtractor(64, 100.0)
    
    print("\n--- STARTING LIVE INFERENCE ---")
    print("You should see the prediction switch when the activity changes.\n")
    
    # Simple sim of 200 frames, sitting --> walking 
    
    for i in range(200):
        # when we hit 100 notify of activity switch
        if i == 100:
            print("\n!!! ACTION: User started WALKING !!!\n")
            sim.set_curr_activity("walking")
            
        # update step
        sim.update()
        
        # signal process, feed to extractor window
        extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())
        features = np.array(extractor.compute_features())
        
        # normalize extracted data to training stats
        feat_norm = (features - train_mean) / train_std
        feat_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0) # Add batch dim
        
        # use model to predict
        with torch.no_grad():
            outputs = model(feat_tensor)
            # Get probabilities (softmax)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            
        pred_label = labels[predicted_class.item()]
        conf_score = confidence.item() * 100
        
        # show results of 10th frame 
        if i % 10 == 0:
            # format strs, extract info from features + current prediction
            print(f"Time: {sim.get_current_time():.2f}s | "
                  f"Freq: {features[2]:.1f}Hz | "
                  f"AI Says: {pred_label} ({conf_score:.1f}%)")
            
            time.sleep(0.05) # slow this down for readability

if __name__ == "__main__":
    run_live_demo()