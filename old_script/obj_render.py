import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import glfw
from PIL import Image

class OBJRenderer:
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.vertices = []
        self.colors = []
        self.faces = []
        self.normals = []
        
    def load_obj(self, filename):
        """Load OBJ file with vertex colors."""
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                
                if values[0] == 'v':
                    # Split vertex line into position and color
                    v = [float(x) for x in values[1:]]
                    self.vertices.append(v[0:3])  # XYZ coordinates
                    self.colors.append(v[3:6])    # RGB values
                elif values[0] == 'f':
                    # Handle face indices
                    face = [int(v.split('/')[0]) - 1 for v in values[1:]]
                    self.faces.append(face)
        
        # Convert to numpy arrays
        self.vertices = np.array(self.vertices, dtype=np.float32)
        self.colors = np.array(self.colors, dtype=np.float32)
        
        # Center and scale the model
        center = (np.max(self.vertices, axis=0) + np.min(self.vertices, axis=0)) / 2
        self.vertices -= center
        scale = np.max(np.abs(self.vertices))
        self.vertices /= scale
        
        # Calculate normals for shading
        self.calculate_normals()
        
    def calculate_normals(self):
        """Calculate vertex normals."""
        self.normals = np.zeros((len(self.vertices), 3), dtype=np.float32)
        
        for face in self.faces:
            v0 = self.vertices[face[0]]
            v1 = self.vertices[face[1]]
            v2 = self.vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            
            self.normals[face[0]] += normal
            self.normals[face[1]] += normal
            self.normals[face[2]] += normal
            
        # Normalize all vertex normals
        norms = np.linalg.norm(self.normals, axis=1)
        norms[norms == 0] = 1
        self.normals = self.normals / norms[:, np.newaxis]
        
    def init_gl(self):
        """Initialize OpenGL context and settings."""
        if not glfw.init():
            return False
        
        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(self.width, self.height, "OBJ Renderer", None, None)
        if not self.window:
            glfw.terminate()
            return False
            
        glfw.make_context_current(self.window)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glViewport(0, 0, self.width, self.height)
        
        return True
        
    def setup_camera(self):
        """Setup camera projection and position."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width/self.height, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -3.0)
        glRotatef(30, 1, 0, 0)    # Tilt for better view
        glRotatef(180, 0, 1, 0)   # 180 degree rotation
        
    def render_rgb(self):
        """Render RGB image of the model using vertex colors."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Disable lighting for direct color rendering
        glDisable(GL_LIGHTING)
        
        self.setup_camera()
        
        # Render model with vertex colors
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex_id in face:
                glColor3fv(self.colors[vertex_id])
                glVertex3fv(self.vertices[vertex_id])
        glEnd()
        
        # Read pixels
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
        rgb_map = np.flip(pixels, axis=0)
        
        return rgb_map
        
    def render_blue(self):
        """Render model in pale blue color."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 1, 1, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        self.setup_camera()
        
        # Set pale blue material properties
        pale_blue = [0.6, 0.7, 0.9]  # Pale blue color
        glMaterialfv(GL_FRONT, GL_AMBIENT, [*pale_blue, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [*pale_blue, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
        
        # Render model
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vertex_id in face:
                glNormal3fv(self.normals[vertex_id])
                glVertex3fv(self.vertices[vertex_id])
        glEnd()
        
        # Read pixels
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
        blue_map = np.flip(pixels, axis=0)
        
        return blue_map
        
    def save_maps(self, obj_path, rgb_path='rgb.png', blue_path='blue.png'):
        """Generate and save RGB and blue renders."""
        if not self.init_gl():
            print("Failed to initialize OpenGL")
            return
            
        self.load_obj(obj_path)
        
        # Generate maps
        rgb_map = self.render_rgb()
        blue_map = self.render_blue()
        
        # Convert to 8-bit format and save
        rgb_map_img = Image.fromarray((rgb_map * 255).astype(np.uint8))
        blue_map_img = Image.fromarray((blue_map * 255).astype(np.uint8))
        
        rgb_map_img.save(rgb_path)
        blue_map_img.save(blue_path)
        
        # Cleanup
        glfw.destroy_window(self.window)
        glfw.terminate()

# Example usage
if __name__ == "__main__":
    renderer = OBJRenderer(width=512, height=512)
    #renderer.save_maps(r'C:\Users\titan\Downloads\tmp0bl8bhm7.obj')
    renderer.save_maps(r'C:\Users\titan\Downloads\tmpvxs5cxfz.obj')