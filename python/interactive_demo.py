import pygame
import torch
import numpy as np
import sensor_sim
import math

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
SIM_RATE = 100.0
WINDOW_SIZE = 256

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # sitting mode
GREEN = (50, 200, 50)   # walking mode
GRAY = (200, 200, 200)

# again prep AI
class ActivityClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 16)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(16, 4)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def load_model():
    checkpoint = torch.load("activity_model.pth", weights_only=False)
    model = ActivityClassifier()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['mean'], checkpoint['std']

# helper to draw stick fig 
def draw_stick_figure(screen, activity, confidence, frame_count):
    cx, cy = 200, 300
    # color coding by activity
    if activity == "SITTING": color = (200, 50, 50)   # RED
    elif activity == "WALKING": color = (50, 200, 50) # GREEN
    elif activity == "RUNNING": color = (50, 50, 200) # BLUE
    elif activity == "JUMPING": color = (200, 200, 50)# YELLOW
    else: color = (0,0,0)
    # animations
    offset_y = 0
    leg_spread = 0
    if activity == "WALKING":
        # slow bob (freq 0.2)
        offset_y = math.sin(frame_count * 0.2) * 10
        leg_spread = math.sin(frame_count * 0.2) * 20
        
    elif activity == "RUNNING":
        # fast bob (freq 0.5), higher knees
        offset_y = math.sin(frame_count * 0.5) * 15
        leg_spread = math.sin(frame_count * 0.5) * 30
        
    elif activity == "JUMPING":
        # manual impulse animation
        # can use modulo to create a periodic jump loop visual
        jump_phase = (frame_count % 60) / 60.0 # 1 second loop
        if jump_phase < 0.2: # Crouch
            offset_y = 10
            leg_spread = 30
        elif jump_phase < 0.5: # Air
            offset_y = -40 # FLY UP!
            leg_spread = 10
        elif jump_phase < 0.6: # Land
            offset_y = 20
            leg_spread = 40
        else: # Recover
            offset_y = 0
            leg_spread = 0

    # head
    pygame.draw.circle(screen, color, (cx, cy - 50 + offset_y), 20, 5)
    # body
    pygame.draw.line(screen, color, (cx, cy - 30 + offset_y), (cx, cy + 50 + offset_y), 5)
    # arms, swing for W/R
    arm_swing = 0
    if activity in ["WALKING", "RUNNING"]:
        arm_swing = math.sin(frame_count * 0.2) * 20
    pygame.draw.line(screen, color, (cx - 30, cy + offset_y - arm_swing), (cx + 30, cy + offset_y + arm_swing), 5)
    # legs
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx - 20 - leg_spread, cy + 100 + offset_y), 5)
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx + 20 + leg_spread, cy + 100 + offset_y), 5)
    # label
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"{activity} ({confidence:.1f}%)", True, color)
    screen.blit(text, (cx - 80, cy + 130))

def draw_ui(screen, true_state, input_features):
    font = pygame.font.SysFont(None, 24)
    # instructions to user
    info = font.render("Press [SPACE] to toggle Simulation Physics", True, BLACK)
    screen.blit(info, (50, 50))
    # ground truth
    state_text = font.render(f"Simulated Physics: {true_state}", True, BLACK)
    screen.blit(state_text, (50, 80))
    # display C++ extracted features
    pygame.draw.rect(screen, GRAY, (450, 100, 300, 400))
    y = 120
    labels = ["Mean Acc Z", "Var Acc Z", "Dom Freq", "Spectral E"]
    title = font.render("Live C++ Features", True, BLACK)
    screen.blit(title, (460, 120))
    for i, label in enumerate(labels):
        val_text = font.render(f"{label}: {input_features[i]:.4f}", True, BLACK)
        screen.blit(val_text, (460, 160 + i*40))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real-Time Activity Recognition Demo")
    clock = pygame.time.Clock()
    
    # prep simulator + extractor + model
    model, train_mean, train_std = load_model()
    sim = sensor_sim.IMUSim(SIM_RATE, "sitting")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SIM_RATE)
    
    true_activity = "sitting"
    frame_count = 0
    
    # initialize UI vars
    pred_label = "SITTING"
    confidence = 0.0
    latest_features = [0,0,0,0]
    
    running = True
    while running:
        # handle inputs 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # cycle of activities
                    if true_activity == "sitting": true_activity = "walking"
                    elif true_activity == "walking": true_activity = "running"
                    elif true_activity == "running": true_activity = "jumping"
                    else: true_activity = "sitting"
                    sim.set_curr_activity(true_activity)
        # run sim, few ticks per second
        for _ in range(2): 
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())

        # extract features, normalize, feed to torch
        # give predictive labels as well
        LABELS = {
            0: "SITTING",
            1: "WALKING",
            2: "RUNNING",
            3: "JUMPING"
        }
        raw_feats = np.array(extractor.compute_features())
        latest_features = raw_feats
        norm_feats = (raw_feats - train_mean) / train_std
        tensor_feats = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_feats)
            probs = torch.softmax(outputs, dim=1)
            conf, cls_idx = torch.max(probs, 1)
            # Use the dictionary to look up the correct name
            idx = cls_idx.item()
            pred_label = LABELS.get(idx, "UNKNOWN")
            confidence = conf.item() * 100
        # rendering
        screen.fill(WHITE)
        draw_stick_figure(screen, pred_label, confidence, frame_count)
        draw_ui(screen, true_activity.upper(), latest_features)
        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()