# Use a lightweight Python base image
FROM python:3.10-slim

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-numpy \
    libeigen3-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    xvfb \
    x11vnc \
    fluxbox \
    net-tools \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Build-Time Python Dependencies
RUN pip install --no-cache-dir pybind11 numpy

# Set up the working directory
WORKDIR /app

# Copy your source code
COPY . /app

# 3. Build the C++ Engine
WORKDIR /app/build
# Calculate pybind11 path dynamically to satisfy CMake
RUN PYBIND11_CMAKE_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())") && \
    cmake .. -Dpybind11_DIR=$PYBIND11_CMAKE_DIR && \
    make

# 4. Install Runtime Python dependencies
WORKDIR /app
RUN pip install --no-cache-dir pygame torch pandas scikit-learn

# 5. Setup Virtual Display Environment
ENV DISPLAY=:0
ENV RESOLUTION=1000x700
# [FIX] Prevent PyGame from crashing looking for a sound card
ENV SDL_AUDIODRIVER=dummy 

# 6. Create the startup script
# [FIX] Switched from 'launch.sh' to direct 'websockify' command
RUN echo '#!/bin/bash\n\
rm -f /tmp/.X0-lock\n\
Xvfb :0 -screen 0 ${RESOLUTION}x24 &\n\
sleep 2\n\
fluxbox &\n\
x11vnc -display :0 -nopw -forever -shared -bg &\n\
websockify --web /usr/share/novnc/ 8080 localhost:5900 &\n\
python3 python/interactive_demo.py\n\
tail -f /dev/null' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Expose the web port
EXPOSE 8080

# Start!
CMD ["/app/entrypoint.sh"]