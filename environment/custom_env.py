import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple, Dict, List, Optional, Any

class AmbulanceEnv(gym.Env):
    """Custom Gymnasium Environment for Ambulance Navigation"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=(10, 10), render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        self.viewer = None
        self.grid_size = grid_size
        self.time_limit = 800
        
        # Car physics parameters
        self.wheel_angle = 0.0
        self.max_wheel_angle = math.radians(30)
        self.car_length = 2.0
        self.drift_factor = 0.95
        
        # Speed control
        self.min_speed = 1.0
        self.max_speed = 8.0
        self.current_speed = 5.0
        self.acceleration = 0.2
        self.braking = 0.3
        
        # Physical dimensions
        self.ambulance_length = 0.8
        self.ambulance_width = 0.4
        
        # Boundaries
        self.min_x = self.ambulance_width/2
        self.max_x = grid_size[0] - self.ambulance_width/2
        self.min_y = self.ambulance_length/2
        self.max_y = grid_size[1] - self.ambulance_length/2
        
        print(f"\nENVIRONMENT BOUNDARIES:")
        print(f"X: {self.min_x:.2f} to {self.max_x:.2f}")
        print(f"Y: {self.min_y:.2f} to {self.max_y:.2f}")
        
        # Initialize elements
        self._setup_fixed_elements()
        
        # Action space
        self.action_space = spaces.Dict({
            "steering": spaces.Discrete(5),  # 0=Straight, 1=Soft-L, 2=Hard-L, 3=Soft-R, 4=Hard-R
            "throttle": spaces.Discrete(3)   # 0=Brake, 1=Maintain, 2=Accelerate
        })
        
        # Observation space
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'direction': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'destination': spaces.Box(low=0, high=max(grid_size), shape=(2,), dtype=np.float32),
            'patient_status': spaces.Discrete(2),
            'time_elapsed': spaces.Box(low=0, high=self.time_limit, shape=(1,), dtype=np.float32),
            'obstacles': spaces.Box(low=0, high=max(grid_size), shape=(5, 2), dtype=np.float32),
            'wheel_angle': spaces.Box(low=-self.max_wheel_angle, high=self.max_wheel_angle, shape=(1,), dtype=np.float32)
        })

    def _setup_fixed_elements(self):
        """Initialize fixed environment elements"""
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

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        
        self.hospitals = self.fixed_hospitals.copy()
        self.patients = self.fixed_patients.copy()
        self.ambulance_pos = np.array(self.fixed_ambulance_start, dtype=np.float32)
        self.ambulance_dir = np.array([0.0, 1.0])
        self.wheel_angle = 0.0
        self.current_speed = 5.0
        self.patient_picked = False
        self.obstacles = [np.array(pos, dtype=np.float32) for pos in self.fixed_obstacles]
        self.current_destination = self.patients[0]
        self.time_elapsed = 0
        self.last_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_observation(), {}

    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        self._take_action(action)
        self.time_elapsed += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Goal rewards
        if not self.patient_picked and self._is_at_location(self.ambulance_pos, [self.current_destination]):
            self.patient_picked = True
            reward += 500
            self.current_destination = self._nearest_hospital()
            self.last_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        elif self.patient_picked and self._is_at_location(self.ambulance_pos, self.hospitals):
            reward += 1000
            terminated = True
        
        # Efficiency reward
        current_distance = self._calculate_distance(self.ambulance_pos, self.current_destination)
        reward += (self.last_distance - current_distance) * 10
        self.last_distance = current_distance
        
        # Penalties
        if self._check_collision():
            reward -= 1000
            terminated = True
        
        reward -= 1  # Time penalty
        
        if self.time_elapsed > self.time_limit:
            truncated = True
        
        if self.render_mode == 'human':
            self.render()
            
        return self._get_observation(), reward, terminated, truncated, {}

    def _take_action(self, action: Dict) -> None:
        steering = action["steering"]
        throttle = action["throttle"]
        
        # Steering logic
        steering_actions = {
            0: 0,        # Straight
            1: 15,       # Soft left
            2: 30,       # Hard left
            3: -15,      # Soft right
            4: -30       # Hard right
        }
        angle_change = math.radians(steering_actions[steering])
        if angle_change != 0:
            self.wheel_angle = np.clip(
                self.wheel_angle + angle_change,
                -self.max_wheel_angle,
                self.max_wheel_angle
            )
        else:
            self.wheel_angle *= self.drift_factor
        
        # Throttle logic
        if throttle == 0:    # Brake
            self.current_speed = max(self.min_speed, self.current_speed - self.braking)
        elif throttle == 2:  # Accelerate
            self.current_speed = min(self.max_speed, self.current_speed + self.acceleration)
        
        # Update position and direction
        if abs(self.wheel_angle) > 0.01:
            turn_radius = self.car_length / math.tan(self.wheel_angle)
            angular_velocity = (self.current_speed * 0.05) / max(0.1, abs(turn_radius))
            angular_velocity *= -1 if self.wheel_angle < 0 else 1
            
            current_angle = math.atan2(self.ambulance_dir[1], self.ambulance_dir[0])
            new_angle = current_angle + angular_velocity
            self.ambulance_dir = np.array([math.cos(new_angle), math.sin(new_angle)])
            self.ambulance_dir /= np.linalg.norm(self.ambulance_dir)
        
        # Calculate new position
        new_pos = self.ambulance_pos + self.ambulance_dir * self.current_speed * 0.05
        
        # Boundary check
        if self._is_position_valid(new_pos):
            self.ambulance_pos = new_pos

    def _is_position_valid(self, pos: np.ndarray) -> bool:
        angle = math.atan2(self.ambulance_dir[1], self.ambulance_dir[0])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        half_len, half_wid = self.ambulance_length/2, self.ambulance_width/2
        
        corners = [
            (pos[0] + half_len*cos_a - half_wid*sin_a, pos[1] + half_len*sin_a + half_wid*cos_a),
            (pos[0] + half_len*cos_a + half_wid*sin_a, pos[1] + half_len*sin_a - half_wid*cos_a),
            (pos[0] - half_len*cos_a - half_wid*sin_a, pos[1] - half_len*sin_a + half_wid*cos_a),
            (pos[0] - half_len*cos_a + half_wid*sin_a, pos[1] - half_len*sin_a - half_wid*cos_a)
        ]
        
        return all(
            self.min_x <= cx <= self.max_x and 
            self.min_y <= cy <= self.max_y 
            for cx, cy in corners
        )

    def _get_observation(self) -> Dict:
        return {
            'position': np.array(self.ambulance_pos, dtype=np.float32),
            'direction': np.array(self.ambulance_dir, dtype=np.float32),
            'destination': np.array(self.current_destination, dtype=np.float32),
            'patient_status': int(self.patient_picked),
            'time_elapsed': np.array([self.time_elapsed], dtype=np.float32),
            'obstacles': np.array(self.obstacles[:5], dtype=np.float32),
            'wheel_angle': np.array([self.wheel_angle], dtype=np.float32)
        }

    def _check_collision(self) -> bool:
        return any(
            self._calculate_distance(self.ambulance_pos, obstacle) < 0.7
            for obstacle in self.obstacles
        )

    def _is_at_location(self, pos: np.ndarray, locations: List[Tuple[float, float]]) -> bool:
        return any(
            self._calculate_distance(pos, loc) < 0.5
            for loc in locations
        )

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)

    def _nearest_hospital(self) -> Tuple[float, float]:
        distances = [self._calculate_distance(self.ambulance_pos, h) for h in self.hospitals]
        return self.hospitals[np.argmin(distances)]

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == 'human':
            if self.viewer is None:
                from rendering import AmbulanceVisualizer
                self.viewer = AmbulanceVisualizer(self)
            return self.viewer.render()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
