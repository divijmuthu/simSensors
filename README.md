# To run:
- You can make use of the interactive_demo.py file to run the full system directly, it'll open up a window that plays out an activity and streams sensor data leading to the classifier recognizing the current activity, and you can cycle through activities with the space bar
- Additionally, if you have Docker Desktop, you can run the full simulation, including the AI and GUI, in your browser without installing any dependencies
  - Using the dockerfile I successfully got a docker image which works and can be used for future deployment, here is the process:
  - Build the image = docker build -t sensor-zoo .
  - Run the container (Access via http://localhost:8080/vnc.html) = docker run -p 8080:8080 sensor-zoo

# Description
This is an exploratory project involving a couple of my interests.
I want to get an idea of the role of various sensor types by simulating their data streams in C++.
In particular, I'm building towards an interactive demo where a user can control a character's activities which are measured by these simulated sensor data streams.
Then these activities can be recognized by an ML model, in this case a neural net in Pytorch. The ML component will probably be enhanced over time with better practices but also 
should be kept lightweight for this real-time classification scenario.

# Overall pipeline
/src contains the C++ files for simulating a stream of sensor data (imu_simulator, older name but captures all sensor streams), extracting relevant features (feature_extractor), and exposing this functionality to Python as a module (pybind_wrapper). On the python side we have generate_dataset to get simulated data as a .csv for training the model, which occurs in train_model, and this preps a PyTorch model which can be used for real time classification in the interactive_demo.
