from __future__ import division

from OpenGL.GL import *
import numpy as np
import time
import math
import json
import sys

from random import uniform, randint

from pyglet import clock, font, image, window
from pyglet.gl import *


WORLD_FILE = "config.json"

# Textures are disabled, replace with your own and uncomment _setup_textures if you want to use them
TEXTURE_FILES = ["ball1.png",
                 "ball2.jpg",
                 "ball3.jpg",
                 "adidas.jpg",
                 "tennisball.jpg",
                 "tennisball2.png"]


TIME_TICK = 0.1
GRAVITY_CONST = 7
GRAVITY_VECTOR = np.array([0, -GRAVITY_CONST, 0])
COLLISON_MARGIN = 0.1
SPACE_LIMITS = [[-200, 200],[-200, 800],[-200, 200]]
DAMPING_FACTOR = 0.5
BALLS = 50


# Helper function to turn arrays into parameters the OpenGl bindings can swallow
def vec(*args):
    return (GLfloat * len(args))(*args)


# Randomly generate objects
def randomize_balls(count):
    data = []
    while len(data) < count:
        ball = {}
        ball['size'] = uniform(20,20)
        ball['mass'] = math.pow(ball['size'], 3)
        ball['pos'] = [uniform(-100,190), uniform(380,230), uniform(0,190)]
        ball['init_vel'] = [uniform(-7,10),uniform(-20,20),uniform(-10,-25)]
        #ball['texture'] = 2
        # Make sure ball doesn't touch any existing ball
        touching = False
        for b in data:
            pos = np.array(ball['pos'])
            pos_b = np.array(b['pos'])
            if np.linalg.norm(pos - pos_b) < (ball['size'] + b['size']):
                touching = True
                
        if not touching:
            data.append(ball)

    return data


# The physical model of an n-dimensional object
class PhysicalBody(object):
    def __init__(self, mass, size, pos, speed):
        self.mass = mass
        self.size = size
        self.pos = np.array(pos)
        self.speed = np.array(speed)
        self.collisions_handled = False

    # true if an object with a given size and posittion is touching this object.
    def is_touching(self, pos, size):
        dist = pos - self.pos
        dist_norm = np.linalg.norm(pos - self.pos)
        return (dist_norm < (size + self.size)) 


    # update this object
    def update_velocity(self):
        self.speed = self.speed + TIME_TICK * GRAVITY_VECTOR

    # optional, call to check for collisions. Returns a list of indexes for collisions
    def get_collisions(self, objects):
        collisions = []
        for n, obj in enumerate(objects):
            if obj != self:
                dist = self.pos - obj.pos
                dist_norm = np.linalg.norm(dist)
                if dist_norm < (self.size + obj.size):
                    collisions.append(n)
        return collisions


    # call after update_velocities to apply the new accelerations and move the object
    def update_pos(self):
        self.pos = self.pos + TIME_TICK * self.speed
        self.handled_collisions = [];

    # randomize an upwards motion    
    def kick(self):
        self.speed += GRAVITY_VECTOR *uniform(2, 8) 

    # resolve collsions with the walls
    def resolve_wall_collisions(self, space_limits):
        for idx, limit in enumerate(space_limits):
            if (self.pos[idx] - self.size) < limit[0]:
                self.speed[idx] *= -(1 - DAMPING_FACTOR)
                self.pos[idx] = limit[0] + self.size + COLLISON_MARGIN;

            elif (self.pos[idx] + self.size) > limit[1]:
                self.speed[idx] *= -(1 - DAMPING_FACTOR)
                self.pos[idx] = limit[1] - self.size - COLLISON_MARGIN;


    # Resolve collisions with other objects
    def resolve_obj_collisions(self, objects):
        for obj in objects:
            if obj != self:
                if obj.is_touching(self.pos, self.size) and not obj.index in self.handled_collisions:
                    direction = self.pos - obj.pos
                    direction = direction / np.linalg.norm(direction)
                    self_dv = np.dot(self.speed, direction)
                    obj_dv = np.dot(obj.speed, direction)
                    acf = obj_dv
                    bcf = self_dv;

                    self.speed += (acf - self_dv) * direction * (1 - DAMPING_FACTOR) * (2 * obj.mass / (self.mass + obj.mass))
                    obj.speed += (bcf - obj_dv) * direction  * (1 - DAMPING_FACTOR)  * (2 * self.mass / (self.mass + obj.mass))

                    self.pos += COLLISON_MARGIN * 3 * direction
                    self.handled_collisions.append(obj.index)
                    obj.handled_collisions.append(self.index)                    
                    pass


class Body3D(PhysicalBody):
    index_counter = 0

    def __init__(self, mass, size, pos = [0, 0, 0], speed = [0, 0, 0]):
        self.mass = mass
        self.size = size
        self.pos = np.array(pos)
        self.speed = np.array(speed)
        self.handled_collisions = []
        self.index = Body3D.index_counter
        Body3D.index_counter += 1


# Extend the physical model with draw functions
class GraphObject(Body3D):
    def __init__(self, mass, size, pos = [0, 0, 0], speed = [0, 0, 0], texture=None):
        super(GraphObject, self).__init__(mass, size, pos, speed)
        self.rotation = 0
        self.texture = texture
        if not self.texture:
            self.r1 = uniform(0.1,1)
            self.r2 = uniform(0.1,1)
            self.g1 = uniform(0.1,1)
            self.g2 = uniform(0.1,1)
            self.b1 = uniform(0.1,1)
            self.b2 = uniform(0.1,1)


    def __draw_sphere(self):
        sphere = gluNewQuadric()
        if self.texture:
            glColor4f(1, 1, 1, 1.0)
            gluQuadricTexture(sphere, GLU_TRUE)
            glEnable(GL_TEXTURE_2D);
            glBindTexture(self.texture.target, self.texture.id)
        else:
            glColor4f(self.r1, self.g1, self.b1, 1.0)
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(self.r1, self.g1, self.b1, 1))

        gluSphere(sphere, 1.0, 24, 24)
        gluDeleteQuadric(sphere)


    def draw(self):     
        glLoadIdentity()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(self.rotation, 0, 0, 1)
        glScalef(self.size, self.size, self.size)

        self.__draw_sphere()


    def update_rot(self):
        self.rotation += 10 / self.size


class World(object):
    def __init__(self, objects, textures):
        self.ticks = 0
        #self.__setup_textures(textures) 
        self.objects = []
        for obj in objects:
            self.objects.append(GraphObject(obj['mass'],
                                            obj['size'],
                                            obj['pos'],
                                            obj['init_vel'] if 'init_vel' in obj else [0.0, 0.0, 0.0],
                                            self.textures[obj['texture']] if 'texture' in obj else None))


    def __setup_textures(self, texture_names):
        self.textures = []
        for file_name in texture_names:
            img = pyglet.image.load(file_name)
            data = img.get_image_data()
            texture = img.get_texture()
            glEnable(texture.target)
            glBindTexture(texture.target, texture.id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 
                    0, GL_RGBA, GL_UNSIGNED_BYTE,
                    img.get_image_data().get_data('RGBA',
                    img.width * 4))
            self.textures.append(texture)


    def tick(self):
        self.ticks +=1
        for obj in self.objects:
            obj.update_velocity()
            #if self.ticks % 300 == 0:
            #    obj.kick()

        for obj in self.objects:
            obj.update_pos()
            obj.update_rot()
            obj.resolve_wall_collisions(SPACE_LIMITS)
            obj.resolve_obj_collisions(self.objects);

        return self.objects[2].pos.tolist()

    def draw(self):
        
        glClearColor(0,0,0,0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW);
        glEnable(GL_LIGHTING)

        for obj in self.objects:
            obj.draw()

        # Draw semi transparent box around the space 
        glDisable(GL_LIGHTING)
        glLoadIdentity();
        glColor4f(0.7,0.7,0.7,0.7)
        
        glRotatef(90,1,0,0)
        glTranslatef(0,0,200)
        glRectf(200,200,-200,-200)

        glColor4f(0.8,0.8,0.8,0.4)
        glRotatef(90,0,1,0)
        glTranslatef(200,0,-200)
        glRectf(200,200,-200,-200)

        glColor4f(0.8,0.8,0.8,0.2)
        glTranslatef(0,0,400)
        glRectf(200,200,-200,-200)

        glColor4f(0.6,0.6,0.6,0.3)
        glRotatef(90,1,0,0)
        glTranslatef(0,-200,200)
        glRectf(200,200,-200,-200)
       


class Camera(object):

    def __init__(self, win, x=0.0, y=0.0, rot=0.0, zoom=1.0):
        self.win = win
        self.x = x
        self.y = y
        self.rot = rot
        self.zoom = zoom
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

    def worldProjection(self, pos = [0,0,0]):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        zwr =  self.zoom * self.win.width / self.win.height
        gluPerspective(50, self.win.width / self.win.height, 0.1, 10000)
        glTranslatef(0, 0, -700)
        glRotatef(-60,0,1,0)
        glRotatef(self.rot,0,1,0)
        self.y -= 0.2
        self.rot +=0.14



class App(object):

    def __init__(self, no_balls, fs = False):
        #world_data = json.loads(open(world_file).read())
        world_data = randomize_balls(BALLS)
        self.world = World(world_data, TEXTURE_FILES)
        self.win = window.Window(fullscreen=fs, vsync=True)
        self.camera = Camera(self.win, zoom=150.0)
        clock.set_fps_limit(60)

    def mainLoop(self):
        # Lighting setup
        glColor3f(1.0, 1.0, 1.0)
        glShadeModel (GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_SPECULAR, vec(0.0, 0.0, 0.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, vec(40.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, vec(1.0, 1.0, 1.0, 0.0))

        glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1, 1 , 1, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION, vec(700, 700, 1800, 0.0))  

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_BLEND)
        glEnable(GL_NORMALIZE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        while not self.win.has_exit:
            self.win.dispatch_events()

            pos = self.world.tick()

            self.camera.worldProjection(pos)
            self.world.draw()

            self.win.flip()


def main():
    # Config from file not implemented yet
    cfg_file = WORLD_FILE;
    fullscreen = False
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
    if len(sys.argv) > 2:
        fullscreen = True if (sys.argv[2].strip() == "fullscreen") else False

    app = App(cfg_file, fullscreen)
    app.mainLoop()



if __name__ == "__main__":
    main()

