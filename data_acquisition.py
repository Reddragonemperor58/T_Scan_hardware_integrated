# --- START OF FILE data_acquisition.py ---
import serial
import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SensorDataReader:
    def __init__(self, port='COM4', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.data = pd.DataFrame(columns=['timestamp', 'tooth_id', 'sensor_point_id', 'force', 'contact_time'])
        self.is_connected = False

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            self.is_connected = True
            logging.info(f"Connected to sensor on {self.port}")
        except serial.SerialException as e:
            logging.error(f"Failed to connect to sensor: {e}")
            self.is_connected = False

    def read_data(self, duration=5):
        if not self.is_connected:
            logging.warning("No sensor connected. Using simulated data.")
            return self.simulate_data(duration=duration, num_teeth=16, num_sensor_points_per_tooth=4) 

        start_time = time.time()
        data_list = []
        while time.time() - start_time < duration:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                if line:
                    try:
                        parts = list(map(float, line.split(',')))
                        if len(parts) == 5: 
                            timestamp, tooth_id, sensor_point_id, force, contact_time = parts
                            data_list.append({
                                'timestamp': timestamp, 'tooth_id': int(tooth_id),
                                'sensor_point_id': int(sensor_point_id), 
                                'force': force, 'contact_time': contact_time
                            })
                        else: logging.warning(f"Invalid data line (expected 5 parts): {line}")
                    except ValueError as e: logging.warning(f"Invalid data format: {line}, error: {e}")
                time.sleep(0.01)
            except serial.SerialException as e: logging.error(f"Serial read error: {e}"); break
        if data_list:
            new_data = pd.DataFrame(data_list)
            self.data = pd.concat([self.data, new_data], ignore_index=True) if not self.data.empty else new_data
        return self.data

    def simulate_data(self, duration=5, num_teeth=16, num_sensor_points_per_tooth=4):
        timestamps = np.arange(0, duration, 0.1)
        data_list = []
        for t in timestamps:
            for tooth_id in range(1, num_teeth + 1):
                base_force_on_tooth = np.random.uniform(5, 60) * (0.8 + 0.4 * np.sin(t * 0.5 + tooth_id * 0.3))
                if 4 <= tooth_id <= 6 or 11 <= tooth_id <= 13: base_force_on_tooth *= np.random.uniform(0.7, 1.3)
                elif tooth_id <= 3 or tooth_id >= 14: base_force_on_tooth *= np.random.uniform(0.9, 1.5)
                for sensor_point_idx in range(1, num_sensor_points_per_tooth + 1):
                    variation_factor = 1.0
                    if num_sensor_points_per_tooth == 4:
                        if sensor_point_idx == 1: variation_factor = np.random.uniform(0.7, 1.1)
                        elif sensor_point_idx == 2: variation_factor = np.random.uniform(0.9, 1.3)
                        elif sensor_point_idx == 3: variation_factor = np.random.uniform(0.6, 1.0)
                        elif sensor_point_idx == 4: variation_factor = np.random.uniform(0.8, 1.2)
                    else: variation_factor = np.random.uniform(0.7, 1.3)
                    force = base_force_on_tooth * variation_factor + np.random.uniform(-10, 10)
                    force = max(0, min(100, force)) 
                    contact_time = np.random.uniform(0.01, 0.05)
                    data_list.append({'timestamp': t, 'tooth_id': tooth_id, 'sensor_point_id': sensor_point_idx,
                                      'force': force, 'contact_time': contact_time})
        sim_data = pd.DataFrame(data_list)
        if not self.data.empty and set(self.data.columns) == set(sim_data.columns):
            self.data = pd.concat([self.data, sim_data], ignore_index=True)
        else:
            if not self.data.empty: logging.warning("Sim data replacing existing data due to column mismatch or empty.")
            self.data = sim_data
        logging.info(f"Generated simulated data: {len(sim_data)} rows, {num_teeth} teeth, {num_sensor_points_per_tooth} sensor points/tooth.")
        return self.data

    def save_data(self, filename='sensor_data.csv'):
        if not self.data.empty: self.data.to_csv(filename, index=False); logging.info(f"Data saved to {filename}")
    def close(self):
        if self.serial and self.is_connected: self.serial.close(); self.is_connected = False; logging.info("Sensor connection closed")
# --- END OF FILE data_acquisition.py ---