import gym
from gym import spaces
import numpy as np
import math
from typing import Tuple, Dict, List

class AmbulanceEnv(gym.Env):
    """Custom Gym Environment for Ambulance Navigation with Dict action space"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=(10, 10)):
        super(AmbulanceEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.time_limit = 800
        
        # Car physics parameters
        self.wheel_angle = 0.0
        self.max_wheel_angle = math.radians(30)
        self.car_length = 2.0
        self.drift_factor = 0.95
        
        # Speed control parameters
        self.min_speed = 1.0
        self.max_speed = 8.0
        self.current_speed = 5.0  # Initial speed matches original base_speed
        self.acceleration = 0.2
        self.braking = 0.3
        
        # Ambulance physical dimensions (unchanged)
        self.ambulance_length = 0.8
        self.ambulance_width = 0.4
        
        # Calculate boundaries (unchanged)
        self.min_x = self.ambulance_width/2
        self.max_x = grid_size[0] - self.ambulance_width/2
        self.min_y = self.ambulance_length/2
        self.max_y = grid_size[1] - self.ambulance_length/2
        
        print(f"\nENVIRONMENT BOUNDARIES:")
        print(f"X: {self.min_x:.2f} to {self.max_x:.2f}")
        print(f"Y: {self.min_y:.2f} to {self.max_y:.2f}")
        
        # Fixed elements setup (EXACTLY as in your original code)
        self._setup_fixed_elements()
        
        # New Dict action space
        self.action_space = spaces.Dict({
            "steering": spaces.Discrete(5),  # 0=Straight, 1=Soft-L, 2=Hard-L, 3=Soft-R, 4=Hard-R
            "throttle": spaces.Discrete(3)   # 0=Brake, 1=Maintain, 2=Accelerate
        })
        
        # Observation space (unchanged)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'destination': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'patient_status': spaces.Discrete(2),
            'time_elapsed': spaces.Box(low=0, high=self.time_limit, shape=(1,), dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=max(grid_size), shape=(5, 2), dtype=np.float32),
            'wheel_angle': spaces.Box(low=-self.max_wheel_angle, high=self.max_wheel_angle, shape=(1,), dtype=np.float32)
        })
        
        self.reset()

    def _setup_fixed_elements(self):
        """Initialize all fixed environment elements (EXACTLY as in your original code)"""
        self.fixed_hospitals = [(2.5, 2.5), (7.5, 7.5)]
        self.fixed_patients = [(3.0, 7.0), (7.0, 3.0), (8.0, 1.0)]
        self.fixed_ambulance_start = (
            np.clip(1.0, self.min_x, self.max_x),
            np.clip(1.0, self.min_y, self.max_y)
        )
        self.fixed_obstacles = [
            (1.5, 6.0), (1.5, 7.0),
            (6.0, 2.0), (7.0, 2.0),
            (5.0, 5.0),
            (5.0, 7.0)
        ]

    def reset(self):
        """Reset to fixed initial state (preserving original positions)"""
        self.hospitals = self.fixed_hospitals.copy()
        self.patients = self.fixed_patients.copy()
        self.ambulance_pos = np.array(self.fixed_ambulance_start, dtype=np.float32)
        self.ambulance_dir = np.array([0.0, 1.0])  # Initial facing direction (up)
        self.wheel_angle = 0.0
        self.current_speed = 5.0  # Matches original base_speed
        self.patient_picked = False
        self.obstacles = [np.array(pos, dtype=np.float32) for pos in self.fixed_obstacles]
        self.current_destination = self.patients[0]
        self.time_elapsed = 0
        self.done = False
        self.last_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        return self._get_observation()

    def step(self, action):
        """Execute one time step with new Dict action handling"""
        if self.done:
            return self._get_observation(), 0, True, {}
        
        self._take_action(action)
        self.time_elapsed += 1
        
        reward = 0
        done = False
        
        # Goal rewards (unchanged logic)
        if not self.patient_picked and self._is_at_location(self.ambulance_pos, [self.current_destination]):
            self.patient_picked = True
            reward += 500
            self.current_destination = self._nearest_hospital()
            self.last_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        elif self.patient_picked and self._is_at_location(self.ambulance_pos, self.hospitals):
            reward += 1000
            done = True
        
        # Efficiency reward (unchanged)
        current_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        reward += (self.last_distance - current_distance) * 10
        self.last_distance = current_distance
        
        # Penalties (unchanged)
        if self._check_collision():
            reward -= 1000
            done = True
        
        reward -= 1  # Time penalty
        
        if self.time_elapsed > self.time_limit:
            reward -= 500
            done = True
        
        self.done = done
        return self._get_observation(), reward, done, {}

    def _take_action(self, action):
        """New movement handling with Dict actions"""
        steering = action["steering"]
        throttle = action["throttle"]
        x, y = self.ambulance_pos
        dir_x, dir_y = self.ambulance_dir
        
        # Handle steering
        if steering == 1:  # Soft left (15°)
            self.wheel_angle = min(self.wheel_angle + math.radians(15), self.max_wheel_angle)
        elif steering == 2:  # Hard left (30°)
            self.wheel_angle = min(self.wheel_angle + math.radians(30), self.max_wheel_angle)
        elif steering == 3:  # Soft right (15°)
            self.wheel_angle = max(self.wheel_angle - math.radians(15), -self.max_wheel_angle)
        elif steering == 4:  # Hard right (30°)
            self.wheel_angle = max(self.wheel_angle - math.radians(30), -self.max_wheel_angle)
        else:  # steering=0 (Straight)
            self.wheel_angle *= self.drift_factor
        
        # Handle throttle
        if throttle == 0:  # Brake
            self.current_speed = max(self.min_speed, self.current_speed - self.braking)
        elif throttle == 2:  # Accelerate
            self.current_speed = min(self.max_speed, self.current_speed + self.acceleration)
        # throttle=1 maintains speed
        
        # Calculate turning (unchanged physics)
        if abs(self.wheel_angle) > 0.01:
            turn_radius = self.car_length / math.tan(self.wheel_angle)
            angular_velocity = (self.current_speed * 0.05) / max(0.1, abs(turn_radius))
            angular_velocity *= -1 if self.wheel_angle < 0 else 1
            
            current_angle = math.atan2(dir_y, dir_x)
            new_angle = current_angle + angular_velocity
            new_dir_x = math.cos(new_angle)
            new_dir_y = math.sin(new_angle)
            
            norm = math.sqrt(new_dir_x**2 + new_dir_y**2)
            self.ambulance_dir = np.array([new_dir_x/norm, new_dir_y/norm])
        
        # Calculate new position with current_speed
        new_x = x + self.ambulance_dir[0] * self.current_speed * 0.05
        new_y = y + self.ambulance_dir[1] * self.current_speed * 0.05
        
        # Boundary check (unchanged from your original)
        angle = math.atan2(dir_y, dir_x)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        half_len = self.ambulance_length/2
        half_wid = self.ambulance_width/2
        
        corners = [
            (new_x + half_len*cos_a - half_wid*sin_a, new_y + half_len*sin_a + half_wid*cos_a),
            (new_x + half_len*cos_a + half_wid*sin_a, new_y + half_len*sin_a - half_wid*cos_a),
            (new_x - half_len*cos_a - half_wid*sin_a, new_y - half_len*sin_a + half_wid*cos_a),
            (new_x - half_len*cos_a + half_wid*sin_a, new_y - half_len*sin_a - half_wid*cos_a)
        ]
        
        # Check all corners
        valid_move = all(
            self.min_x <= cx <= self.max_x and 
            self.min_y <= cy <= self.max_y 
            for cx, cy in corners
        )
        
        if valid_move:
            self.ambulance_pos = (new_x, new_y)

    # Keep all your original helper methods exactly as they were:
    def _check_collision(self):
        for obstacle in self.obstacles:
            if self._calculate_distance(self.ambulance_pos, obstacle) < 0.7:
                return True
        return False

    def _is_at_location(self, pos, locations):
        for loc in locations:
            if self._calculate_distance(pos, loc) < 0.5:
                return True
        return False

    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

    def _nearest_hospital(self):
        min_dist = float('inf')
        nearest = self.hospitals[0]
        for hospital in self.hospitals:
            dist = self._calculate_distance(self.ambulance_pos, hospital)
            if dist < min_dist:
                min_dist = dist
                nearest = hospital
        return nearest

    def _get_observation(self):
        return {
            'position': np.array(self.ambulance_pos, dtype=np.float32),
            'direction': np.array(self.ambulance_dir, dtype=np.float32),
            'destination': np.array(self.current_destination, dtype=np.float32),
            'patient_status': int(self.patient_picked),
            'time_elapsed': np.array([self.time_elapsed], dtype=np.float32),
            'obstacles': np.array(self.obstacles[:5], dtype=np.float32),
            'wheel_angle': np.array([self.wheel_angle], dtype=np.float32)
        }

    def render(self, mode='human'):
        if mode == 'human':
            from rendering2 import AmbulanceVisualizer
            if not hasattr(self, 'viewer'):
                self.viewer = AmbulanceVisualizer(self)
            return self.viewer.render()

    def close(self):
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
