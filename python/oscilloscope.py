# python/oscilloscope.py
import pygame
import sensor_sim
import numpy as np

# Config
WIDTH, HEIGHT = 800, 600
SIM_RATE = 100.0
WINDOW_SIZE = 256 # Match your C++ window size

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
DARK_GRAY = (50, 50, 50)

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("C++ Edge-AI Oscilloscope")
    clock = pygame.time.Clock()

    # C++ Pipeline
    sim = sensor_sim.IMUSim(SIM_RATE, "walking")
    extractor = sensor_sim.FeatureExtractor(WINDOW_SIZE, SIM_RATE)

    running = True
    while running:
        for event in pygame.event.get():
           if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    sim.set_curr_activity("sitting")
                    print("Activity: SITTING")
                elif event.key == pygame.K_2:
                    sim.set_curr_activity("walking")
                    print("Activity: WALKING")
                elif event.key == pygame.K_3:
                    sim.set_curr_activity("running")
                    print("Activity: RUNNING")
                elif event.key == pygame.K_4:
                    sim.set_curr_activity("jumping")
                    print("Activity: JUMPING")

        # 1. Update Physics (Fast Stride)
        for _ in range(2):
            sim.update()
            extractor.add_sample(sim.get_acceleration(), sim.get_gyroscope())

        # 2. Get Data from C++
        raw_wave = np.array(extractor.get_z_accel_buffer())
        fft_data = np.array(extractor.get_fft_spectrum())

        # 3. Draw
        screen.fill(BLACK)

        # --- TOP GRAPH: Raw Accelerometer Z ---
        # Draw Axis
        pygame.draw.line(screen, DARK_GRAY, (0, 150), (WIDTH, 150), 1)
        
        if len(raw_wave) > 1:
            points = []
            for i, val in enumerate(raw_wave):
                x = map_range(i, 0, len(raw_wave), 0, WIDTH)
                # Map 8.0g~12.0g to screen height 0~300
                y = map_range(val, 0, 40, 300, 0) 
                points.append((x, y))
            pygame.draw.lines(screen, GREEN, False, points, 2)

        # --- BOTTOM GRAPH: FFT Spectrum ---
        # Draw Axis
        pygame.draw.line(screen, DARK_GRAY, (0, 600), (WIDTH, 600), 1)
        
        if len(fft_data) > 1:
            bar_width = WIDTH / len(fft_data)
            for i, val in enumerate(fft_data):
                # Ignore DC component (index 0) usually huge
                if i == 0: continue
                
                x = i * bar_width
                # Scale magnitude to height (approx 0-100)
                height = min(val * 2, 300) 
                pygame.draw.rect(screen, CYAN, (x, HEIGHT - height, bar_width - 1, height))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()