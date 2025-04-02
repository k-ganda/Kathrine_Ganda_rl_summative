import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import time
import numpy as np

class AmbulanceVisualizer:
    def __init__(self, env):
        self.env = env
        self.window_width = 1280
        self.window_height = 720
        self.camera_distance = 20
        self.camera_angle = 45
        self.camera_height = 15
        self.last_time = 0
        self.fps = 0
        self.frame_count = 0
        self.wheel_rotation = 0
        self.light_flash = False
        self.light_timer = 0
        self.fixed_camera = True
        self.show_patient_marker = True
        self.smoke_particles = []
        self.mission_status = ""
        self.status_timer = 0
        self.patient_in_ambulance = False
        self.patient_transition = 0.0
        
        self._init_opengl()
        
    def _init_opengl(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow(b"Ambulance Mission Simulator")
        
        glutDisplayFunc(self._render)
        glutIdleFunc(self._update)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutSpecialFunc(self._special_keys)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        
        # Additional OpenGL settings for better visibility
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearDepth(1.0)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        
        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        
        # Emergency light setup
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [1, 0, 0, 1])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [1, 0.3, 0.3, 1])
        glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 30.0)
        glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 2.0)
        
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1, 1, 1, 1])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50])

    def _reshape(self, width, height):
        self.window_width = width
        self.window_height = height
        glViewport(0, 0, width, height)
        self._setup_camera()
        
    def _keyboard(self, key, x, y):
        if key == b'\x1b':  # ESC to exit
            glutDestroyWindow(glutGetWindow())
        elif key == b'f' or key == b'F':
            self.fixed_camera = not self.fixed_camera
        elif key == b'd' or key == b'D':
            self.show_patient_marker = not self.show_patient_marker

    def _special_keys(self, key, x, y):
        mod = glutGetModifiers()
        if mod & GLUT_ACTIVE_CTRL:
            if key == GLUT_KEY_UP:
                self.camera_distance = max(10, self.camera_distance - 1)
            elif key == GLUT_KEY_DOWN:
                self.camera_distance = min(50, self.camera_distance + 1)
        elif key == GLUT_KEY_LEFT:
            self.camera_angle = (self.camera_angle + 5) % 360
        elif key == GLUT_KEY_RIGHT:
            self.camera_angle = (self.camera_angle - 5) % 360
                
    def _update(self):
        current_time = glutGet(GLUT_ELAPSED_TIME) / 1000.0
        time_elapsed = current_time - self.last_time
        
        # FPS calculation
        self.frame_count += 1
        if time_elapsed >= 1.0:
            self.fps = self.frame_count / time_elapsed
            self.frame_count = 0
            self.last_time = current_time
            
        # Update patient transition
        if self.env.patient_picked and not self.patient_in_ambulance:
            self.patient_transition = min(1.0, self.patient_transition + time_elapsed * 2)
            if self.patient_transition >= 1.0:
                self.patient_in_ambulance = True
        elif not self.env.patient_picked and self.patient_in_ambulance:
            self.patient_transition = max(0.0, self.patient_transition - time_elapsed * 2)
            if self.patient_transition <= 0.0:
                self.patient_in_ambulance = False
            
        # Enhanced wheel rotation physics
        speed = self.env.current_speed
        wheel_circumference = 2.0  # Approximate wheel circumference in meters
        distance_per_rotation = wheel_circumference
        rotations_per_second = speed / distance_per_rotation
        degrees_per_second = rotations_per_second * 360
        
        self.wheel_rotation = (self.wheel_rotation + degrees_per_second * time_elapsed) % 360
        
        # Update smoke particles for brake/acceleration effects
        self._update_smoke_particles(time_elapsed)
            
        # Emergency light flashing
        self.light_timer += time_elapsed
        if self.light_timer > 0.25:
            self.light_flash = not self.light_flash
            self.light_timer = 0
            
        # Update mission status timer
        if self.mission_status:
            self.status_timer += time_elapsed
            if self.status_timer > 3.0:  # Show status for 3 seconds
                self.mission_status = ""
                self.status_timer = 0
                
        glutPostRedisplay()
        
    def _update_smoke_particles(self, dt):
        """Update particle effects for braking/acceleration"""
        new_particles = []
        for particle in self.smoke_particles:
            # Update particle position and lifetime
            particle['position'][0] += particle['velocity'][0] * dt
            particle['position'][1] += particle['velocity'][1] * dt
            particle['position'][2] += particle['velocity'][2] * dt
            particle['lifetime'] -= dt
            
            if particle['lifetime'] > 0:
                new_particles.append(particle)
        
        self.smoke_particles = new_particles
        
        # Add new particles if braking or accelerating
        if self.env.current_speed < 2.0:  # Braking effect
            self._add_brake_particles()
        elif self.env.current_speed > 6.0:  # Acceleration effect
            self._add_acceleration_particles()
            
    def _add_brake_particles(self):
        """Add particles for braking effect"""
        for _ in range(2):  # Fewer particles for braking
            self.smoke_particles.append({
                'position': [-0.4 + random.uniform(-0.1, 0.1), 
                             random.choice([-0.25, 0.25]), 
                             -0.2],
                'velocity': [random.uniform(-0.5, 0.5), 
                            random.uniform(-0.1, 0.1), 
                            random.uniform(0.1, 0.3)],
                'color': [0.8, 0.1, 0.1, 0.8],  # Red brake dust
                'size': random.uniform(0.05, 0.1),
                'lifetime': random.uniform(0.5, 1.0)
            })
            
    def _add_acceleration_particles(self):
        """Add particles for acceleration effect"""
        for _ in range(5):  # More particles for acceleration
            self.smoke_particles.append({
                'position': [-0.5 + random.uniform(-0.1, 0.1), 
                             random.choice([-0.25, 0.25]), 
                             -0.2],
                'velocity': [random.uniform(-1.0, -0.5),  # Backward velocity
                            random.uniform(-0.2, 0.2), 
                            random.uniform(0.1, 0.5)],
                'color': [0.3, 0.3, 0.3, 0.6],  # Gray exhaust smoke
                'size': random.uniform(0.1, 0.15),
                'lifetime': random.uniform(0.8, 1.5)
            })
        
    def _setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.window_width/self.window_height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        
    def render(self):
        glutMainLoopEvent()
        self._render()
        
    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Camera setup
        if self.fixed_camera:
            center_x = self.env.grid_size[0] / 2
            center_y = self.env.grid_size[1] / 2
            gluLookAt(center_x, center_y, self.camera_height,
                     center_x, center_y, 0,
                     0, 1, 0)
        else:
            amb_x, amb_y = self.env.ambulance_pos
            cam_x = amb_x - math.sin(math.radians(self.camera_angle)) * self.camera_distance
            cam_y = amb_y - math.cos(math.radians(self.camera_angle)) * self.camera_distance
            gluLookAt(cam_x, cam_y, self.camera_height,
                     amb_x, amb_y, 0,
                     0, 0, 1)
        
        # Emergency lights
        if self.light_flash:
            glEnable(GL_LIGHT1)
            amb_x, amb_y = self.env.ambulance_pos
            glLightfv(GL_LIGHT1, GL_POSITION, [amb_x, amb_y, 2.5, 1])
            glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, [0, 0, -1])
        else:
            glDisable(GL_LIGHT1)
        
        # Draw all elements
        self._draw_ground()
        self._draw_road_network()
        self._draw_buildings()
        self._draw_traffic()
        self._draw_patients()
        self._draw_ambulance()
        self._draw_boundary_markers()
        
        
        glutSwapBuffers()
    
    def _draw_ground(self):
        """Draw the ground plane"""
        glColor3f(0.3, 0.5, 0.2)  # Grass color
        min_x, max_x = self.env.min_x, self.env.max_x
        min_y, max_y = self.env.min_y, self.env.max_y
        
        # Add some padding around the environment
        padding = 2.0
        glBegin(GL_QUADS)
        glNormal3f(0, 0, 1)
        glVertex3f(min_x - padding, min_y - padding, -0.1)
        glVertex3f(max_x + padding, min_y - padding, -0.1)
        glVertex3f(max_x + padding, max_y + padding, -0.1)
        glVertex3f(min_x - padding, max_y + padding, -0.1)
        glEnd()

    def _draw_road_network(self):
        """Draw road grid with properly extended lane markings"""
        # Draw roads first (darker color for better contrast)
        glColor3f(0.15, 0.15, 0.15)
        min_x, max_x = self.env.min_x, self.env.max_x
        min_y, max_y = self.env.min_y, self.env.max_y
        
        # Horizontal roads (full width)
        for y in np.arange(0, self.env.grid_size[1] + 1, 1):
            glBegin(GL_QUADS)
            glNormal3f(0, 0, 1)
            glVertex3f(min_x, y - 0.5, 0.01)
            glVertex3f(max_x, y - 0.5, 0.01)
            glVertex3f(max_x, y + 0.5, 0.01)
            glVertex3f(min_x, y + 0.5, 0.01)
            glEnd()

        # Vertical roads (full height)
        for x in np.arange(0, self.env.grid_size[0] + 1, 1):
            glBegin(GL_QUADS)
            glNormal3f(0, 0, 1)
            glVertex3f(x - 0.5, min_y, 0.01)
            glVertex3f(x + 0.5, min_y, 0.01)
            glVertex3f(x + 0.5, max_y, 0.01)
            glVertex3f(x - 0.5, max_y, 0.01)
            glEnd()

        # Draw lane markings with enhanced visibility
        glDisable(GL_LIGHTING)
        glDepthFunc(GL_LEQUAL)
        glColor4f(1, 1, 1, 0.95)  # Bright white with slight transparency
        
        # Horizontal center lines (full width)
        for y in np.arange(0.5, self.env.grid_size[1], 1):
            self._draw_center_line((min_x, y), (max_x, y))
        
        # Vertical center lines (full height)
        for x in np.arange(0.5, self.env.grid_size[0], 1):
            self._draw_center_line((x, min_y), (x, max_y))
        
        glEnable(GL_LIGHTING)
        glDepthFunc(GL_LESS)

    def _draw_center_line(self, start_pos, end_pos):
        """Draw a continuous dashed center line between two points"""
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Calculate line direction and length
        dx = end_x - start_x
        dy = end_y - start_y
        line_length = math.sqrt(dx*dx + dy*dy)
        dir_x = dx/line_length
        dir_y = dy/line_length
        
        # Line parameters
        dash_length = 0.3
        gap_length = 0.15
        line_width = 0.05
        z_height = 0.05
        
        # Draw dashes along the line
        current_dist = 0
        while current_dist < line_length:
            dash_start = (start_x + dir_x * current_dist, 
                        start_y + dir_y * current_dist)
            dash_end_dist = min(current_dist + dash_length, line_length)
            dash_end = (start_x + dir_x * dash_end_dist,
                        start_y + dir_y * dash_end_dist)
            
            # Calculate perpendicular direction for line width
            perp_x = -dir_y * line_width
            perp_y = dir_x * line_width
            
            # Draw the dash segment
            glBegin(GL_QUADS)
            glVertex3f(dash_start[0] - perp_x, dash_start[1] - perp_y, z_height)
            glVertex3f(dash_start[0] + perp_x, dash_start[1] + perp_y, z_height)
            glVertex3f(dash_end[0] + perp_x, dash_end[1] + perp_y, z_height)
            glVertex3f(dash_end[0] - perp_x, dash_end[1] - perp_y, z_height)
            glEnd()
            
            current_dist += dash_length + gap_length

    def _draw_boundary_markers(self):
        """Draw red boundaries matching environment limits"""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        min_x, max_x = self.env.min_x, self.env.max_x
        min_y, max_y = self.env.min_y, self.env.max_y
        
        # Boundary rectangle
        glColor3f(1, 0, 0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(min_x, min_y, 0.1)
        glVertex3f(max_x, min_y, 0.1)
        glVertex3f(max_x, max_y, 0.1)
        glVertex3f(min_x, max_y, 0.1)
        glEnd()
        
        # Corner markers (alternating colors)
        for i, (cx, cy) in enumerate([(min_x, min_y), (max_x, min_y), 
                                     (max_x, max_y), (min_x, max_y)]):
            glPushMatrix()
            glTranslatef(cx, cy, 0.1)
            glColor3f(1, 0.5, 0) if i % 2 else glColor3f(0, 1, 1)
            glutSolidSphere(0.2, 16, 16)
            glPopMatrix()
        
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)

    def _draw_buildings(self):
        """Draw hospitals with indicators for delivery points"""
        for i, hospital in enumerate(self.env.hospitals):
            self._draw_hospital(hospital, i)
            
    def _draw_hospital(self, pos, index):
        """Draw a properly scaled and aligned hospital"""
        x, y = pos
        glPushMatrix()
        
        # Align to grid properly
        aligned_x = math.floor(x) + 0.5 if x % 1 > 0.5 else math.ceil(x) - 0.5
        aligned_y = math.floor(y) + 0.5 if y % 1 > 0.5 else math.ceil(y) - 0.5
        glTranslatef(aligned_x, aligned_y, 0)
        
        # Scaled down main building (original size * 0.7)
        glColor3f(0.9, 0.9, 1.0)
        glPushMatrix()
        glScalef(1.05, 1.05, 2.1)  # 1.5 * 0.7 = 1.05, 3 * 0.7 = 2.1
        glutSolidCube(1)
        glPopMatrix()
        
        # Scaled down red cross
        glColor3f(1, 0, 0)
        glPushMatrix()
        glTranslatef(0, 0, 2.1)  
        glScalef(0.7, 0.14, 0.14)  
        glutSolidCube(1)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(0, 0, 2.1)  
        glScalef(0.14, 0.7, 0.14)  
        glutSolidCube(1)
        glPopMatrix()
        
        # Scaled down windows
        glColor4f(0.7, 0.8, 1.0, 0.7)
        glPushMatrix()
        glTranslatef(0, -0.56, 1.05) 
        glScalef(0.56, 0.07, 1.4)  
        glutSolidCube(1)
        glPopMatrix()
        
        
        
        glPopMatrix()

    def _draw_traffic(self):
        """Draw obstacle cars"""
        for obstacle in self.env.obstacles:
            self._draw_car(obstacle)
            
    def _draw_car(self, pos):
        """Draw a single traffic car"""
        x, y = pos
        glPushMatrix()
        glTranslatef(x, y, 0.3)
        
        # Orient along nearest road
        if abs(x - round(x)) > abs(y - round(y)):  # Horizontal road
            glRotatef(90, 0, 0, 1)
        
        # Car body
        glColor3f(0.8, 0.1, 0.1)  # Red
        glPushMatrix()
        glScalef(0.8, 0.4, 0.3)
        glutSolidCube(1)
        glPopMatrix()
        
        # Roof
        glPushMatrix()
        glTranslatef(0, 0, 0.3)
        glScalef(0.7, 0.35, 0.2)
        glutSolidCube(1)
        glPopMatrix()
        
        # Windows
        glColor4f(0.7, 0.8, 1.0, 0.5)
        glPushMatrix()
        glTranslatef(0, 0, 0.45)
        glScalef(0.6, 0.3, 0.01)
        glutSolidCube(1)
        glPopMatrix()
        
        # Wheels
        glColor3f(0.1, 0.1, 0.1)
        for wx, wy in [(0.35,0.25), (0.35,-0.25), (-0.35,0.25), (-0.35,-0.25)]:
            glPushMatrix()
            glTranslatef(wx, wy, -0.15)
            glRotatef(90, 0, 1, 0)
            glutSolidTorus(0.05, 0.1, 10, 16)
            
            # Wheel hub
            glColor3f(0.5, 0.5, 0.5)
            glutSolidTorus(0.02, 0.08, 8, 16)
            glColor3f(0.1, 0.1, 0.1)
            glPopMatrix()
        
        glPopMatrix()

    def _draw_patients(self):
        """Draw only active patients with target highlighting"""
        for i, patient in enumerate(self.env.active_patients):
            is_target = (not self.env.patient_picked and 
                        np.allclose(patient, self.env.current_destination))
            self._draw_patient(patient, is_target, i)
            
            # Draw the pulsing marker for current target
            if is_target and self.show_patient_marker:
                self._draw_patient_marker(patient, i)

    def _draw_patient(self, pos, is_target=False, patient_id=0):
        """Draw a single patient"""
        x, y = pos
        glPushMatrix()
        glTranslatef(x, y, 0.3)
        
        # Stretcher
        glColor3f(0.9, 0.9, 0.1) if is_target else glColor3f(0.5, 0.5, 0.5)
        glPushMatrix()
        glScalef(0.8, 0.3, 0.05)
        glutSolidCube(1)
        glPopMatrix()
        
        # Patient
        glColor3f(1.0, 0.9, 0.8)  # Skin tone
        glPushMatrix()
        glTranslatef(0, 0, 0.4)
        glutSolidSphere(0.15, 16, 16)  # Head
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(0, 0, 0.25)
        glScalef(0.4, 0.2, 0.3)
        glutSolidCube(1)  # Torso
        glPopMatrix()
        
        # Patient ID badge
        glDisable(GL_LIGHTING)
        glColor3f(0, 0, 1)
        glPushMatrix()
        glTranslatef(0.2, 0, 0.35)
        glScalef(0.1, 0.1, 0.01)
        glutSolidCube(1)
        glPopMatrix()
        glEnable(GL_LIGHTING)
        
        glPopMatrix()

    def _draw_patient_marker(self, patient, patient_id):
        """Pulsing marker above target patient"""
        try:
            x, y = patient
            pulse = math.sin(time.time() * 5) * 0.1 + 0.5
            
            glDisable(GL_LIGHTING)
            glColor3f(1, 1, 0)  # Yellow
            glPushMatrix()
            glTranslatef(x, y, 2)
            glutSolidSphere(0.3 * pulse, 16, 16)
            glPopMatrix()
            
            # Marker line
            glBegin(GL_LINES)
            glVertex3f(x, y, 2)
            glVertex3f(x, y, 0.4)
            glEnd()
            
            # Text label
            glRasterPos3f(x, y, 2.5)
            text = f"Patient {patient_id+1}"
            for char in text:
                glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
            
            glEnable(GL_LIGHTING)
        except:
            pass
        
    def _draw_ambulance(self):
        """Draw the ambulance with proper steering and patient visualization"""
        x, y = self.env.ambulance_pos
        dir_x, dir_y = self.env.ambulance_dir
        
        # Calculate orientation
        body_angle = math.degrees(math.atan2(dir_y, dir_x))
        wheel_angle_deg = math.degrees(self.env.wheel_angle)
        
        glPushMatrix()
        glTranslatef(x, y, 0.3)
        glRotatef(body_angle, 0, 0, 1)

        # Main body
        speed_factor = min(1.0, (self.env.current_speed - self.env.min_speed) / 
                         (self.env.max_speed - self.env.min_speed))
        glColor3f(1, 1 - speed_factor*0.3, 1 - speed_factor*0.3)  # White to slight red tint at high speed
        glPushMatrix()
        glScalef(0.8, 0.4, 0.5)
        glutSolidCube(1)
        glPopMatrix()

        # Cabin
        glPushMatrix()
        glTranslatef(-0.15, 0, 0.25)
        glScalef(0.6, 0.4, 0.4)
        glutSolidCube(1)
        glPopMatrix()

        # Roof with lights
        if self.light_flash:
            glColor3f(1, 0, 0)  # Bright red
        else:
            glColor3f(0.7, 0, 0)  # Dark red
        glPushMatrix()
        glTranslatef(0, 0, 0.6)
        glScalef(0.7, 0.5, 0.15)
        glutSolidCube(1)
        glPopMatrix()

        # Stripes
        glColor3f(1, 0, 0)
        for y_offset in [0.25, -0.25]:
            glPushMatrix()
            glTranslatef(0, y_offset, 0.4)
            glScalef(0.8, 0.05, 0.05)
            glutSolidCube(1)
            glPopMatrix()

        # Wheels
        glColor3f(0.1, 0.1, 0.1)  # Tires
        # Front wheels (steerable)
        for x_pos, y_pos in [(0.4, 0.2), (0.4, -0.2)]:
            glPushMatrix()
            glTranslatef(x_pos, y_pos, -0.25)
            glRotatef(wheel_angle_deg, 0, 0, 1)  # Steering
            glRotatef(self.wheel_rotation, 0, 1, 0)  # Rolling
            glRotatef(90, 0, 1, 0)
            glutSolidTorus(0.05, 0.1, 12, 24)
            
            # Hub
            glColor3f(0.5, 0.5, 0.5)
            glutSolidTorus(0.02, 0.08, 8, 16)
            glColor3f(0.1, 0.1, 0.1)
            glPopMatrix()
        
        # Rear wheels (fixed)
        for x_pos, y_pos in [(-0.4, 0.2), (-0.4, -0.2)]:
            glPushMatrix()
            glTranslatef(x_pos, y_pos, -0.25)
            glRotatef(self.wheel_rotation, 0, 1, 0)
            glRotatef(90, 0, 1, 0)
            glutSolidTorus(0.05, 0.1, 12, 24)
            
            # Hub
            glColor3f(0.5, 0.5, 0.5)
            glutSolidTorus(0.02, 0.08, 8, 16)
            glPopMatrix()
            
        # Draw patient in ambulance (if picked up)
        if self.patient_transition > 0:
            glPushMatrix()
            glTranslatef(0, 0, 0.5 * self.patient_transition)
            
            # Stretcher
            glColor3f(0.5, 0.5, 0.5)
            glPushMatrix()
            glScalef(0.7, 0.25, 0.04)
            glutSolidCube(1)
            glPopMatrix()
            
            # Patient
            glColor3f(1.0, 0.9, 0.8)  # Skin tone
            glPushMatrix()
            glTranslatef(0, 0, 0.1)
            glutSolidSphere(0.12, 16, 16)  # Head
            glPopMatrix()
            
            glPushMatrix()
            glTranslatef(0, 0, 0.05)
            glScalef(0.35, 0.15, 0.2)
            glutSolidCube(1)  # Torso
            glPopMatrix()
            
            glPopMatrix()
            
        # Draw smoke particles
        self._draw_smoke_particles()
    
        glPopMatrix()  # End ambulance transform
        
    def _draw_smoke_particles(self):
        """Draw all active smoke particles"""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        
        for particle in self.smoke_particles:
            alpha = particle['color'][3] * (particle['lifetime'] / 1.0)  # Fade out
            glColor4f(particle['color'][0], particle['color'][1], 
                     particle['color'][2], alpha)
            
            glPushMatrix()
            glTranslatef(*particle['position'])
            glutSolidSphere(particle['size'] * (1.0 + (1.0 - particle['lifetime'])), 
                          8, 8)  # Particles grow as they fade
            glPopMatrix()
        
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        
        
    def close(self):
        """Cleanup"""
        glutDestroyWindow(glutGetWindow())
