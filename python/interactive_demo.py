import pygame
import torch
import numpy as np
import sensor_sim
import math

# --- CONFIGURATION ---
WINDOW_WIDTH = 900  # Wider for dashboard
WINDOW_HEIGHT = 700 # Taller for split screen
FPS = 60
SIM_RATE = 100.0
WINDOW_SIZE = 256

# Dimensions
WORLD_HEIGHT = 500
DASHBOARD_HEIGHT = 200

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # sitting
GREEN = (50, 200, 50)   # walking
BLUE = (50, 50, 200)    # running
YELLOW = (200, 200, 50) # jumping
CYAN = (50, 200, 200)   # stairs
MAGENTA = (200, 50, 200)# elevator
GRAY = (220, 220, 220)  # Dashboard background
DARK_GRAY = (50, 50, 50)
BRIGHT_BLUE = (0, 100, 255)

# --- AI CONFIGURATION ---
class ActivityClassifier(torch.nn.Module):
    def __init__(self):
        super(ActivityClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(5, 64)
        self.relu = torch.nn.ReLU()
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

# --- VISUALIZATION HELPERS ---

def draw_environment_cues(screen, activity, cx, cy, frame_count):
    if activity == "ELEVATOR":
        rect_color = (100, 100, 100)
        pygame.draw.rect(screen, rect_color, (cx - 60, cy - 100, 120, 240), 5)
        arrow_y = (frame_count * 2) % 50
        pygame.draw.polygon(screen, rect_color, [(cx + 80, cy - 20 - arrow_y), (cx + 90, cy - arrow_y), (cx + 70, cy - arrow_y)])

    elif activity == "STAIRS":
        step_color = CYAN
        start_x, start_y = cx - 100, cy + 120
        points = []
        for i in range(5):
            points.append((start_x + i*40, start_y - i*40))
            points.append((start_x + (i+1)*40, start_y - i*40))
        pygame.draw.lines(screen, step_color, False, points, 5)

def draw_stick_figure(screen, activity, confidence, frame_count, cx, cy):
    # Draw context
    draw_environment_cues(screen, activity, cx, cy, frame_count)

    # Color Selection
    colors = {
        "SITTING": RED, "WALKING": GREEN, "RUNNING": BLUE,
        "JUMPING": YELLOW, "STAIRS": CYAN, "ELEVATOR": MAGENTA
    }
    color = colors.get(activity, BLACK)

    # Animation Physics
    offset_y = 0
    leg_spread = 0
    
    if activity in ["WALKING", "STAIRS"]:
        offset_y = math.sin(frame_count * 0.2) * 10
        leg_spread = math.sin(frame_count * 0.2) * 20
    elif activity == "RUNNING":
        offset_y = math.sin(frame_count * 0.5) * 15
        leg_spread = math.sin(frame_count * 0.5) * 30
    elif activity == "JUMPING":
        jump_phase = (frame_count % 60) / 60.0
        if jump_phase < 0.2: offset_y, leg_spread = 10, 30
        elif jump_phase < 0.5: offset_y, leg_spread = -40, 10
        elif jump_phase < 0.6: offset_y, leg_spread = 20, 40
        else: offset_y, leg_spread = 0, 0
    elif activity == "ELEVATOR":
        offset_y = math.sin(frame_count * 0.8) * 1

    # Drawing
    pygame.draw.circle(screen, color, (cx, cy - 50 + offset_y), 20, 5) # Head
    pygame.draw.line(screen, color, (cx, cy - 30 + offset_y), (cx, cy + 50 + offset_y), 5) # Body
    
    # Arms
    arm_swing = 0
    if activity in ["WALKING", "RUNNING", "STAIRS"]:
        arm_swing = math.sin(frame_count * 0.2) * 20
    pygame.draw.line(screen, color, (cx - 30, cy + offset_y - arm_swing), (cx + 30, cy + offset_y + arm_swing), 5)
    
    # Legs
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx - 20 - leg_spread, cy + 100 + offset_y), 5)
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx + 20 + leg_spread, cy + 100 + offset_y), 5)

    # Prediction Label (Above Head)
    font = pygame.font.SysFont(None, 36)
    text = font.render(f"AI: {activity} ({confidence:.0f}%)", True, color)
    screen.blit(text, (cx - 60, cy - 90))

def draw_wall_and_sensor(screen, cx, cy, sim_distance):
    # 1. Draw the Wall (Big Red Block)
    wall_x = 800
    pygame.draw.rect(screen, (150, 50, 50), (wall_x, 50, 50, WORLD_HEIGHT - 50))
    
    # 2. Visualize Sensor Cone
    sensor_origin = (cx + 20, cy)
    pixel_dist = sim_distance * 100 # 1m = 100px
    if pixel_dist > (wall_x - cx): pixel_dist = wall_x - cx

    # Color Logic
    if sim_distance >= 4.0:
        cone_color = (200, 200, 200, 50) # Out of range
    else:
        alpha = max(50, 255 - int(sim_distance * 60))
        cone_color = (0, 255, 0, alpha)

    # Draw Beam
    s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    beam_end_x = sensor_origin[0] + pixel_dist
    beam_spread = pixel_dist * 0.2
    points = [sensor_origin, (beam_end_x, sensor_origin[1] - beam_spread), (beam_end_x, sensor_origin[1] + beam_spread)]
    pygame.draw.polygon(s, cone_color, points)
    screen.blit(s, (0,0))

def draw_dashboard(screen, true_state, input_features, tof_val):
    # Background
    pygame.draw.rect(screen, GRAY, (0, WORLD_HEIGHT, WINDOW_WIDTH, DASHBOARD_HEIGHT))
    pygame.draw.line(screen, DARK_GRAY, (0, WORLD_HEIGHT), (WINDOW_WIDTH, WORLD_HEIGHT), 3)

    font = pygame.font.SysFont(None, 24)
    font_bold = pygame.font.SysFont(None, 30)
    title_font = pygame.font.SysFont(None, 36)

    # --- COLUMN 1: CONTROLS & STATE ---
    pygame.draw.rect(screen, WHITE, (20, WORLD_HEIGHT + 20, 250, 160), border_radius=10)
    screen.blit(title_font.render("System Control", True, BLACK), (30, WORLD_HEIGHT + 30))
    
    screen.blit(font.render("SPACE: Toggle Activity", True, DARK_GRAY), (30, WORLD_HEIGHT + 70))
    screen.blit(font.render("ARROWS: Move Character", True, DARK_GRAY), (30, WORLD_HEIGHT + 95))
    
    # Active State Box
    pygame.draw.rect(screen, (230, 240, 255), (30, WORLD_HEIGHT + 130, 230, 40), border_radius=5)
    screen.blit(font.render("Input State:", True, BLACK), (40, WORLD_HEIGHT + 142))
    screen.blit(font_bold.render(true_state, True, BRIGHT_BLUE), (140, WORLD_HEIGHT + 140))

    # --- COLUMN 2: AI FEATURES (The "Why") ---
    pygame.draw.rect(screen, WHITE, (290, WORLD_HEIGHT + 20, 280, 160), border_radius=10)
    screen.blit(title_font.render("Feature Extraction", True, BLACK), (300, WORLD_HEIGHT + 30))
    
    labels = ["Mean Acc Z", "Var Acc Z", "Dom Freq", "Spec Energy", "Vert Vel"]
    for i, label in enumerate(labels):
        val_str = f"{input_features[i]:.4f}"
        y_pos = WORLD_HEIGHT + 70 + (i * 22)
        screen.blit(font.render(label, True, DARK_GRAY), (300, y_pos))
        screen.blit(font.render(val_str, True, BLACK), (450, y_pos))

    # --- COLUMN 3: RAW SENSORS (The "What") ---
    pygame.draw.rect(screen, WHITE, (590, WORLD_HEIGHT + 20, 280, 160), border_radius=10)
    screen.blit(title_font.render("Raw Sensor Data", True, BLACK), (600, WORLD_HEIGHT + 30))
    
    # ToF Sensor
    screen.blit(font_bold.render("ToF / Proximity:", True, DARK_GRAY), (600, WORLD_HEIGHT + 70))
    tof_color = GREEN if tof_val < 4.0 else RED
    screen.blit(font_bold.render(f"{tof_val:.2f} m", True, tof_color), (760, WORLD_HEIGHT + 70))

    # Placeholder for future sensors (Mag, GPS)
    screen.blit(font.render("Barometer: Active", True, DARK_GRAY), (600, WORLD_HEIGHT + 100))
    screen.blit(font.render("Magnetometer: --", True, GRAY), (600, WORLD_HEIGHT + 125))


# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sensor Zoo: Integrated Dashboard")
    clock = pygame.time.Clock()
    
    model, train_mean, train_std = load_model()
    sim = sensor_sim.IMUSim(SIM_RATE, "sitting")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SIM_RATE)
    
    true_activity = "sitting"
    frame_count = 0
    char_x = 100 # Start further left
    char_speed = 0
    
    LABELS = {0: "SITTING", 1: "WALKING", 2: "RUNNING", 3: "JUMPING", 4: "STAIRS", 5: "ELEVATOR"}
    ACT_LIST = ["sitting", "walking", "running", "jumping", "stairs_up", "elevator_up"]
    act_idx = 0
    
    pred_label = "SITTING"
    confidence = 0.0
    latest_features = [0]*5
    
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
                if event.key == pygame.K_RIGHT: char_speed = 3.0
                if event.key == pygame.K_LEFT: char_speed = -3.0
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_RIGHT, pygame.K_LEFT]: char_speed = 0

        # Physics & Movement
        char_x += char_speed
        if char_x < 50: char_x = 50
        if char_x > 750: char_x = 750 # Stop before wall
        
        # Calculate Wall Distance
        dist_meters = (800 - char_x) / 100.0 # Wall is at 800
        sim.set_obstacle_distance(dist_meters)

        # Sim Steps
        for _ in range(2): 
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope(), sim.get_pressure())

        # Inference
        raw_feats = np.array(extractor.compute_features())
        norm_feats = (raw_feats - train_mean) / train_std
        tensor_feats = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_feats)
            probs = torch.softmax(outputs, dim=1)
            conf, cls_idx = torch.max(probs, 1)
            pred_label = LABELS.get(cls_idx.item(), "UNKNOWN")
            confidence = conf.item() * 100

        # Draw Frame
        screen.fill(WHITE)
        
        # World View (Top 500px)
        noisy_dist = sim.get_proximity()
        draw_wall_and_sensor(screen, char_x, 250, noisy_dist) # Wall & Cone
        draw_stick_figure(screen, pred_label, confidence, frame_count, char_x, 250) # Avatar
        
        # Dashboard View (Bottom 200px)
        draw_dashboard(screen, true_activity.upper(), raw_feats, noisy_dist)
        
        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()