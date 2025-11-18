# python/test.py
import sensor_sim  # This imports your .so file!
import time

print("--- Python: Attempting to import C++ module ---")

# We can now use our C++ class AS IF it were a Python class
# We pass "EECS Student" to the C++ constructor
sim = sensor_sim.IMUSim("EECS Student")

print("--- Python: Created C++ object 'sim' ---")
cpp_message = sensor_sim.helloEverybody()

print(f"--- Python: Received message from C++! ---")
print(f"    Message: {cpp_message}")