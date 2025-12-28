import pygame
import torch
import numpy as np
import sensor_sim
import math
import time

# config
WINDOW_WIDTH = 1000  
WINDOW_HEIGHT = 700 
FPS = 60
SIM_RATE = 100.0
WINDOW_SIZE = 256

# Dimensions
WORLD_HEIGHT = 450
DASHBOARD_HEIGHT = 250

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)     # sitting
GREEN = (50, 200, 50)   # walking
BLUE = (50, 50, 200)    # running
YELLOW = (200, 200, 50) # jumping
CYAN = (50, 200, 200)   # stairs
MAGENTA = (200, 50, 200)# elevator
GRAY = (230, 230, 230)  
DARK_GRAY = (50, 50, 50)
BRIGHT_BLUE = (0, 100, 255)
ALERT_ORANGE = (255, 100, 0)

# --- AI CONFIGURATION ---
class ActivityClassifier(torch.nn.Module):
    def __init__(self):
        super(ActivityClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(7, 128) 
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(128, 64) 
        self.output = torch.nn.Linear(64, 6)
        
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

# --- EXPLAINABILITY HELPER ---
def get_ai_reasoning(pred_label, features):
    # features: [MeanAz, VarAz, Freq, Energy, VertVel, MagX, MagY]
    if pred_label == "SITTING": return "Low Variance & Energy"
    elif pred_label in ["WALKING", "RUNNING"]: return f"Periodic Motion ({features[2]:.1f} Hz)"
    elif pred_label == "JUMPING": return "High Impact Energy"
    elif pred_label == "STAIRS": return "Vert Vel + Forward Motion" # Updated Reason
    elif pred_label == "ELEVATOR": return "Pure Vertical Velocity"
    return "Analyzing..."

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
    draw_environment_cues(screen, activity, cx, cy, frame_count)
    colors = {"SITTING": RED, "WALKING": GREEN, "RUNNING": BLUE, "JUMPING": YELLOW, "STAIRS": CYAN, "ELEVATOR": MAGENTA}
    color = colors.get(activity, BLACK)

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

    pygame.draw.circle(screen, color, (cx, cy - 50 + offset_y), 20, 5) 
    pygame.draw.line(screen, color, (cx, cy - 30 + offset_y), (cx, cy + 50 + offset_y), 5) 
    
    arm_swing = 0
    if activity in ["WALKING", "RUNNING", "STAIRS"]:
        arm_swing = math.sin(frame_count * 0.2) * 20
    pygame.draw.line(screen, color, (cx - 30, cy + offset_y - arm_swing), (cx + 30, cy + offset_y + arm_swing), 5)
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx - 20 - leg_spread, cy + 100 + offset_y), 5)
    pygame.draw.line(screen, color, (cx, cy + 50 + offset_y), (cx + 20 + leg_spread, cy + 100 + offset_y), 5)

    font = pygame.font.SysFont(None, 36)
    text = font.render(f"AI: {activity} ({confidence:.0f}%)", True, color)
    screen.blit(text, (cx - 60, cy - 90))

def draw_wall_and_sensor(screen, cx, cy, sim_distance):
    wall_x = 900
    pygame.draw.rect(screen, (150, 50, 50), (wall_x - 50, 50, 50, WORLD_HEIGHT - 50))
    sensor_origin = (cx + 20, cy)
    pixel_dist = sim_distance * 100 
    if pixel_dist > (wall_x - 50 - cx): pixel_dist = wall_x - 50 - cx

    if sim_distance >= 8.0:
        cone_color = (200, 200, 200, 50) 
    else:
        alpha = max(30, 255 - int(sim_distance * 30))
        cone_color = (0, 255, 0, alpha)

    s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    beam_end_x = sensor_origin[0] + pixel_dist
    beam_spread = pixel_dist * 0.2
    points = [sensor_origin, (beam_end_x, sensor_origin[1] - beam_spread), (beam_end_x, sensor_origin[1] + beam_spread)]
    pygame.draw.polygon(s, cone_color, points)
    screen.blit(s, (0,0))

def draw_dashboard(screen, true_state, pred_state, input_features, tof_val, gps_lat, gps_lon, mag_heading, pressure, metrics, char_speed):
    pygame.draw.rect(screen, GRAY, (0, WORLD_HEIGHT, WINDOW_WIDTH, DASHBOARD_HEIGHT))
    pygame.draw.line(screen, DARK_GRAY, (0, WORLD_HEIGHT), (WINDOW_WIDTH, WORLD_HEIGHT), 3)

    font = pygame.font.SysFont(None, 24)
    font_bold = pygame.font.SysFont(None, 28)
    title_font = pygame.font.SysFont(None, 32)

    # --- COL 1: METRICS ---
    pygame.draw.rect(screen, WHITE, (20, WORLD_HEIGHT + 20, 280, 210), border_radius=10)
    screen.blit(title_font.render("System Benchmarks", True, BLACK), (30, WORLD_HEIGHT + 30))
    
    # Updated Labels
    labels = ["C++ Sim Time (us)", "ML Latency (ms)", "Demo FPS"]
    values = [f"{metrics['sim_time']*1e6:.1f}", f"{metrics['latency']*1000:.2f}", f"{metrics['fps']:.1f}"]
    
    for i, (lab, val) in enumerate(zip(labels, values)):
        y = WORLD_HEIGHT + 70 + (i * 30)
        screen.blit(font.render(lab, True, DARK_GRAY), (30, y))
        screen.blit(font_bold.render(val, True, BLACK), (200, y))
    
    # Running Accuracy
    screen.blit(font.render("Running Accuracy:", True, DARK_GRAY), (30, WORLD_HEIGHT + 170))
    acc_color = GREEN if metrics['accuracy'] > 90 else (RED if metrics['accuracy'] < 70 else BLACK)
    screen.blit(font_bold.render(f"{metrics['accuracy']:.1f}%", True, acc_color), (200, WORLD_HEIGHT + 170))

    # --- COL 2: EXPLAINABLE AI ---
    pygame.draw.rect(screen, WHITE, (320, WORLD_HEIGHT + 20, 320, 210), border_radius=10)
    screen.blit(title_font.render("Explainable AI", True, BLACK), (330, WORLD_HEIGHT + 30))
    
    # Input State & Warning Logic
    pygame.draw.rect(screen, (230, 240, 255), (330, WORLD_HEIGHT + 65, 290, 35), border_radius=5)
    screen.blit(font.render("Mode:", True, BLACK), (340, WORLD_HEIGHT + 75))
    screen.blit(font_bold.render(true_state, True, BRIGHT_BLUE), (400, WORLD_HEIGHT + 73))
    
    # [NEW] MOVEMENT WARNING
    # If mode implies movement (Walk/Run/Stairs) but speed is 0
    if true_state in ["WALKING", "RUNNING", "STAIRS_UP"] and abs(char_speed) < 0.1:
        if (pygame.time.get_ticks() % 1000) < 500: # Blink
            warn_font = pygame.font.SysFont(None, 24)
            warn_text = warn_font.render("ACTION REQUIRED: WALK!", True, ALERT_ORANGE)
            screen.blit(warn_text, (400, WORLD_HEIGHT + 105))
    
    # Top Reason
    reason = get_ai_reasoning(pred_state, input_features)
    screen.blit(font.render("Decision Logic:", True, DARK_GRAY), (330, WORLD_HEIGHT + 130))
    screen.blit(font_bold.render(reason, True, BRIGHT_BLUE), (330, WORLD_HEIGHT + 155))

    # --- COL 3: SENSOR ZOO ---
    pygame.draw.rect(screen, WHITE, (660, WORLD_HEIGHT + 20, 320, 210), border_radius=10)
    screen.blit(title_font.render("Sensor Array", True, BLACK), (670, WORLD_HEIGHT + 30))
    
    # ToF
    screen.blit(font_bold.render("ToF / LiDAR:", True, DARK_GRAY), (670, WORLD_HEIGHT + 70))
    tof_color = GREEN if tof_val < 7.9 else RED
    screen.blit(font_bold.render(f"{tof_val:.2f} m", True, tof_color), (800, WORLD_HEIGHT + 70))

    # GPS
    screen.blit(font.render(f"GPS: {gps_lat:.4f}, {gps_lon:.4f}", True, BLACK), (670, WORLD_HEIGHT + 100))
    
    # Mag
    heading_deg = math.degrees(math.atan2(mag_heading[1], mag_heading[0]))
    heading_str = "NORTH"
    if 45 < heading_deg < 135: heading_str = "EAST"
    elif -135 < heading_deg < -45: heading_str = "WEST"
    elif abs(heading_deg) > 135: heading_str = "SOUTH"
    screen.blit(font.render(f"Compass: {heading_str}", True, BLACK), (670, WORLD_HEIGHT + 125))
    
    # Barometer
    pressure_hpa = pressure / 100.0
    screen.blit(font.render(f"Baro: {pressure_hpa:.1f} hPa", True, BLACK), (670, WORLD_HEIGHT + 150))


# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sensor Zoo: Final Engineering Demo")
    clock = pygame.time.Clock()
    
    model, train_mean, train_std = load_model()
    sim = sensor_sim.IMUSim(SIM_RATE, "sitting")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SIM_RATE)
    
    true_activity = "sitting"
    frame_count = 0
    char_x = 100 
    char_speed = 0
    
    LABELS = {0: "SITTING", 1: "WALKING", 2: "RUNNING", 3: "JUMPING", 4: "STAIRS", 5: "ELEVATOR"}
    ACT_LIST = ["sitting", "walking", "running", "jumping", "stairs_up", "elevator_up"]
    act_idx = 0
    
    pred_label = "SITTING"
    confidence = 0.0
    latest_features = [0]*7
    
    total_frames = 0
    correct_frames = 0
    metrics = {"fps": 0, "latency": 0, "accuracy": 0, "sim_time": 0}
    
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

        # Enforce Logic: Elevator = No Walking
        if true_activity == "elevator_up":
            char_speed = 0

        char_x += char_speed
        if char_x < 50: char_x = 50
        if char_x > 850: char_x = 850 
        
        dist_meters = (900 - char_x) / 100.0
        sim.set_obstacle_distance(dist_meters)
        sim.set_velocity(char_speed * 0.5)

        # [BENCHMARK] Measure C++ Sim Time
        t_sim_start = time.perf_counter()
        for _ in range(2): 
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope(), sim.get_magnetometer(), sim.get_pressure())
        t_sim_end = time.perf_counter()
        metrics["sim_time"] = (t_sim_end - t_sim_start) / 2.0 # Average per tick

        # Inference
        raw_feats = np.array(extractor.compute_features())
        t_infer_start = time.perf_counter()
        norm_feats = (raw_feats - train_mean) / train_std
        tensor_feats = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor_feats)
            probs = torch.softmax(outputs, dim=1)
            conf, cls_idx = torch.max(probs, 1)
            pred_label = LABELS.get(cls_idx.item(), "UNKNOWN")
            confidence = conf.item() * 100
        metrics["latency"] = time.perf_counter() - t_infer_start

        if confidence > 80.0:
            total_frames += 1
            if pred_label.lower().replace(" ", "") in true_activity.lower().replace("_up", ""):
                correct_frames += 1
            elif "stairs" in true_activity and "stairs" in pred_label.lower():
                correct_frames += 1
        
        if total_frames > 0:
            metrics["accuracy"] = (correct_frames / total_frames) * 100

        screen.fill(WHITE)
        noisy_dist = sim.get_proximity()
        gps_lat = sim.get_latitude()
        gps_lon = sim.get_longitude()
        mag_head = sim.get_magnetometer() 
        pressure = sim.get_pressure()
        
        draw_wall_and_sensor(screen, char_x, 250, noisy_dist)
        draw_stick_figure(screen, pred_label, confidence, frame_count, char_x, 250)
        
        # Pass char_speed so dashboard can warn user if stationary
        draw_dashboard(screen, true_activity.upper(), pred_label, raw_feats, 
                       noisy_dist, gps_lat, gps_lon, mag_head, pressure, metrics, char_speed)
        
        pygame.display.flip()
        clock.tick(FPS)
        metrics["fps"] = clock.get_fps()
        frame_count += 1

    pygame.quit()

if __name__ == "__main__":
    main()