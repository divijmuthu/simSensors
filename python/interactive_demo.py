import pygame
import torch
import numpy as np
import sensor_sim
import math

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
SIM_RATE = 100.0
WINDOW_SIZE = 256 # match your C++ and Training

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # sitting
GREEN = (50, 200, 50)   # walking
BLUE = (50, 50, 200)    # running
YELLOW = (200, 200, 50) # jumping
CYAN = (50, 200, 200)   # stairs
MAGENTA = (200, 50, 200)# elevator
GRAY = (200, 200, 200)
BRIGHT_BLUE = (0, 100, 255)

# AI config, aligns with train_model.py
class ActivityClassifier(torch.nn.Module):
    def __init__(self):
        super(ActivityClassifier, self).__init__()
        # Upgrade to 32 neurons (matches the Brain Upgrade)
        self.layer1 = torch.nn.Linear(5, 64)
        self.relu = torch.nn.ReLU()
        # Note: Dropout is not needed for Inference/Eval
        self.layer2 = torch.nn.Linear(64, 32)
        self.output = torch.nn.Linear(32, 6)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def load_model():
    checkpoint = torch.load("activity_model.pth", weights_only=False)
    model = ActivityClassifier()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['mean'], checkpoint['std']

def draw_environment_cues(screen, activity, cx, cy, frame_count):
    if activity == "ELEVATOR":
        # Draw the "Box"
        rect_color = (100, 100, 100) # Dark Gray
        # Box around the player
        pygame.draw.rect(screen, rect_color, (cx - 60, cy - 100, 120, 240), 5)
        
        # Draw Up/Down Arrows moving
        arrow_y = (frame_count * 2) % 50
        pygame.draw.polygon(screen, rect_color, [(cx + 80, cy - 20 - arrow_y), (cx + 90, cy - arrow_y), (cx + 70, cy - arrow_y)])

    elif activity == "STAIRS":
        # Draw jagged steps
        step_color = (50, 200, 200) # Cyan
        start_x, start_y = cx - 100, cy + 120
        points = []
        for i in range(5):
            points.append((start_x + i*40, start_y - i*40))
            points.append((start_x + (i+1)*40, start_y - i*40))
        pygame.draw.lines(screen, step_color, False, points, 5)

def draw_stick_figure(screen, activity, confidence, frame_count, cx, cy): # <--- Added cx, cy args
    # [NEW] Draw context first (so it's behind the person)
    # Pass the dynamic cx, cy to the environment drawer too
    draw_environment_cues(screen, activity, cx, cy, frame_count)

    # Color Coding
    if activity == "SITTING": color = RED
    elif activity == "WALKING": color = GREEN
    elif activity == "RUNNING": color = BLUE
    elif activity == "JUMPING": color = YELLOW
    elif activity == "STAIRS": color = CYAN
    elif activity == "ELEVATOR": color = MAGENTA
    else: color = BLACK

    # Animation Physics
    offset_y = 0
    leg_spread = 0
    
    if activity in ["WALKING", "STAIRS"]:
        # Slow bob
        offset_y = math.sin(frame_count * 0.2) * 10
        leg_spread = math.sin(frame_count * 0.2) * 20
        
    elif activity == "RUNNING":
        # Fast bob
        offset_y = math.sin(frame_count * 0.5) * 15
        leg_spread = math.sin(frame_count * 0.5) * 30
        
    elif activity == "JUMPING":
        jump_phase = (frame_count % 60) / 60.0
        if jump_phase < 0.2: offset_y, leg_spread = 10, 30
        elif jump_phase < 0.5: offset_y, leg_spread = -40, 10
        elif jump_phase < 0.6: offset_y, leg_spread = 20, 40
        else: offset_y, leg_spread = 0, 0
    
    elif activity == "ELEVATOR":
        # Standing still, maybe subtle vibration?
        offset_y = math.sin(frame_count * 0.8) * 1 # Tiny engine vibration
        leg_spread = 0

    # Draw Head
    pygame.draw.circle(screen, color, (cx, cy - 50 + offset_y), 20, 5)
    # Draw Body
    pygame.draw.line(screen, color, (cx, cy - 30 + offset_y), (cx, cy + 50 + offset_y), 5)
    
    # Arms
    arm_swing = 0
    if activity in ["WALKING", "RUNNING", "STAIRS"]:
        arm_swing = math.sin(frame_count * 0.2) * 20
    pygame.draw.line(screen, color, (cx - 30, cy + offset_y - arm_swing), (cx + 30, cy + offset_y + arm_swing), 5)
    
    # Legs
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx - 20 - leg_spread, cy + 100 + offset_y), 5)
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx + 20 + leg_spread, cy + 100 + offset_y), 5)

    # Label
    font = pygame.font.SysFont(None, 40)
    text = font.render(f"AI: {activity} ({confidence:.1f}%)", True, color)
    screen.blit(text, (cx - 80, cy + 130))

def draw_ui(screen, true_state, input_features):
    font = pygame.font.SysFont(None, 24)
    font_bold = pygame.font.SysFont(None, 32)
    
    # Instructions
    info = font.render("Press [SPACE] to cycle Physics", True, BLACK)
    screen.blit(info, (50, 30))
    
    # Ground Truth (Make this POP so user knows their input worked)
    pygame.draw.rect(screen, (240, 240, 255), (40, 60, 300, 40)) # Light blue box
    state_label = font.render("Simulated Input:", True, BLACK)
    state_val = font_bold.render(f"{true_state}", True, BRIGHT_BLUE)
    screen.blit(state_label, (50, 70))
    screen.blit(state_val, (200, 70))
    
    # Live Features Box
    pygame.draw.rect(screen, GRAY, (450, 100, 300, 400))
    
    title = font.render("Live C++ Features", True, BLACK)
    screen.blit(title, (460, 120))
    
    labels = ["Mean Acc Z", "Var Acc Z", "Dom Freq", "Energy", "Vert Vel"]
    for i, label in enumerate(labels):
        val_text = font.render(f"{label}: {input_features[i]:.4f}", True, BLACK)
        screen.blit(val_text, (460, 160 + i*40))

def draw_wall_and_sensor(screen, cx, cy, sim_distance):
    # 1. Draw the Wall (Fixed at x=700)
    wall_x = 700
    pygame.draw.rect(screen, (100, 50, 50), (wall_x, 100, 50, 400))
    
    # 2. Visualize the Sensor Cone
    # Origin is the Stick Figure's chest
    sensor_origin = (cx + 20, cy) 
    
    # Visual Range (Map 4 meters to pixels)
    # Let's say 1 meter = 100 pixels
    pixel_dist = sim_distance * 100
    
    # Clamp for visuals
    if pixel_dist > (wall_x - cx): pixel_dist = wall_x - cx

    # Sensor Status Color
    if sim_distance >= 4.0:
        cone_color = (200, 200, 200, 50) # Faint Gray (Out of Range)
    else:
        # Green intensity fades with distance
        alpha = max(50, 255 - int(sim_distance * 60))
        cone_color = (0, 255, 0, alpha)

    # Draw "Beam" (Triangle/Cone)
    # Enable transparency
    s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    
    beam_end_x = sensor_origin[0] + pixel_dist
    beam_y_top = sensor_origin[1] - (pixel_dist * 0.2) # Spread
    beam_y_bot = sensor_origin[1] + (pixel_dist * 0.2)
    
    points = [sensor_origin, (beam_end_x, beam_y_top), (beam_end_x, beam_y_bot)]
    pygame.draw.polygon(s, cone_color, points)
    screen.blit(s, (0,0))
    
    # 3. Draw the Text Value
    font = pygame.font.SysFont(None, 24)
    # Show the NOISY reading from C++
    label = font.render(f"ToF Sensor: {sim_distance:.2f}m", True, BLACK)
    screen.blit(label, (cx + 20, cy - 40))

# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Real-Time Activity Recognition Demo")
    clock = pygame.time.Clock()
    
    # 1. Initialize Pipeline
    model, train_mean, train_std = load_model()
    sim = sensor_sim.IMUSim(SIM_RATE, "sitting")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SIM_RATE)
    
    true_activity = "sitting"
    frame_count = 0

    char_x = 200 # Starting X position
    char_speed = 0
    
    # UI Variables
    pred_label = "SITTING"
    confidence = 0.0
    latest_features = [0,0,0,0,0]
    
    # Label Map
    # Label Map (Matches generate_dataset.py)
    LABELS = {
        0: "SITTING", 1: "WALKING", 2: "RUNNING", 
        3: "JUMPING", 4: "STAIRS", 5: "ELEVATOR"
    }
    
    ACT_LIST = ["sitting", "walking", "running", "jumping", "stairs_up", "elevator_up"]
    act_idx = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    act_idx = (act_idx + 1) % len(ACT_LIST)
                    true_activity = ACT_LIST[act_idx]
                    sim.set_curr_activity(true_activity)
                
                # [NEW] Manual Movement Controls for the "Game" feel
                if event.key == pygame.K_RIGHT:
                    char_speed = 2.0
                if event.key == pygame.K_LEFT:
                    char_speed = -2.0
            
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_RIGHT, pygame.K_LEFT]:
                    char_speed = 0

        # update char position
        char_x += char_speed
        if char_x < 50: char_x = 50
        if char_x > 650: char_x = 650 # Stop before wall
        # compute wall dist data to send to sim
        dist_meters = (700 - char_x) / 100.0
        sim.set_obstacle_distance(dist_meters)

        for _ in range(2): 
            sim.update()
            # [UPDATED] Pass Pressure
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope(), sim.get_pressure())

        raw_feats = np.array(extractor.compute_features())
        latest_features = raw_feats
        
        norm_feats = (raw_feats - train_mean) / train_std
        tensor_feats = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_feats)
            probs = torch.softmax(outputs, dim=1)
            conf, cls_idx = torch.max(probs, 1)
            
            idx = cls_idx.item()
            pred_label = LABELS.get(idx, "UNKNOWN")
            confidence = conf.item() * 100

        screen.fill(WHITE)
        noisy_dist = sim.get_proximity()
        draw_wall_and_sensor(screen, char_x, 300, noisy_dist)
        draw_stick_figure(screen, pred_label, confidence, frame_count, char_x, 300)
        draw_ui(screen, true_activity.upper(), latest_features)
        
        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()