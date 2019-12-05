import random
import math
import numpy as np
import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions as draw

LENGTH_COEFF = 6
f=0.1
y_outer=5100*f
x_outer=8100*f
f2 = 0.1 # factor to multiply by all dimensions given in rules manual
x1 = (x_outer-8100*f2)/2# bottom left
y1 = (y_outer-5100*f2)/2
x2 = x1+8100*f2 #bottom right
y2 = y1
x3 = x1 #top left
y3 = y1+5100*f2
x4 = x2 #top right
y4 = y3

goal=False
prev_dist=0.0
# PyGame init
width = 810
height = 510
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

car = pygame.image.load('car.png').convert_alpha()
car=pygame.transform.scale(car, (100, 100))
red = pygame.image.load('Red.png').convert_alpha()
red=pygame.transform.rotate(red, 90)
blue = pygame.image.load('Blue.png').convert_alpha()
blue=pygame.transform.rotate(blue, -90)
# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = False


class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False
        self.drawoptions = draw(screen)
        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        #self.space.add_collision_handler(1, 1, post_solve=self.car_crashed)
        #pymunk.CollisionHandler(_handler=self.new_handle,post_solve=self.car_crashed,space = self.space)
        # Create the car.
        self.create_car(50, 50, 0)
        # Record steps.
        self.time = 0
        self.num_steps = 0
        self.goal = (400, 51)
        #self.create_obstacle(self.goal[0],self.goal[1],30)
        # screen.blit(car, ())
        screen.blit(blue, (0, 0))
        screen.blit(blue, (0, 410))
        x, y = self.car_body.position
        print(x, y)
        self.init_heuristic = Vec2d(self.goal[0]-x, self.goal[1]-y).get_length()
        self.prev_goal_distance = self.init_heuristic
        self.car_body_prev_angle = 0
        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['green']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        o1x =  50
        o1y =  12.5+385
        
        global prev_dist
        
        prev_dist = math.sqrt((self.goal[0]-100)*(self.goal[0]-100)+(510-self.goal[1]-100)*(510-self.goal[1]-100))
        
        self.obstacles.append(self.create_rect_obstacle(o1x,o1y,100,25))
        self.obstacles.append(self.create_rect_obstacle(400,112.5,100,25))
        self.obstacles.append(self.create_rect_obstacle(400,o1y,100,25))
        self.obstacles.append(self.create_rect_obstacle(760,112.5,100,25))
        
        self.obstacles.append(self.create_rect_obstacle(610,255,80,25))
        self.obstacles.append(self.create_rect_obstacle(200,255,80,25))
        
        self.obstacles.append(self.create_rect_obstacle(150+12.5,50,25,100))
        self.obstacles.append(self.create_rect_obstacle(810-(150+12.5),510-50,25,100))
        
        self.obstacles.append(self.center_obstacle())
        
        self.create_buff_debuff((0, 255, 0), 400, 51, 54, 48)# buff zone
        # self.create_buff_debuff("green", 400, 51, 54, 48)# buff zone
        self.obstacles.append(self.create_debuff( 190, 193.5, 54, 48))#debuff zone
        self.create_buff_debuff( (255, 255, 0), 50, 336, 54, 48)#supply zone
        self.create_buff_debuff((255, 255, 0), 400, 510-51, 54, 48)#supply zone
        self.obstacles.append(self.create_debuff( 810-190, 510-193.5, 54, 48))#debuff
        self.create_buff_debuff((0, 0, 255), 810-50, 510-336, 54, 48)#enemy buff

    def new_handle(self):
        pass
    
    def create_buff_debuff(self, color, x, y, w, h):
        pygame.draw.rect(screen, color, (x-(w/2), 510-(y+h/2), w, h))
        
    def create_debuff(self, x, y, w, h):
        brick_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        brick_body.position = x, y
        brick_shape = pymunk.Poly.create_box(brick_body, (w,h))
        brick_shape.elasticity = 1.0
        brick_shape.color = THECOLORS['red']
        # brick_body.color = THECOLORS['black']
        brick_shape.collision_type = 1
        
        self.space.add(brick_body, brick_shape)
        # brick_shape.position = x, 510-y
        return brick_shape
    def center_obstacle(self):
        brick_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        brick_body.position = width/2, height/2
        points= [(0, 21.21), (-21.21, 0), (0, -21.21), (21.21, 0)]
        brick_shape = pymunk.Poly(brick_body, points)
        brick_shape.elasticity = 1.0
        # brick_shape.position = width/2, height/2
        brick_shape.color = THECOLORS['black']
        brick_shape.collision_type = 1
        self.space.add(brick_body, brick_shape)
        return brick_shape
    
    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(100,100)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_shape.collision_type = 1
        c_body.position = x, y
        c_shape.color = THECOLORS["black"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_rect_obstacle(self, x, y, w, h):
        brick_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        brick_body.position = x, y
        brick_shape = pymunk.Poly.create_box(brick_body, (w,h))
        brick_shape.elasticity = 1.0
        brick_shape.color = THECOLORS['black']
        # brick_body.color = THECOLORS['black']
        brick_shape.collision_type = 1
        
        self.space.add(brick_body, brick_shape)
        # brick_shape.position = x, 510-y
        return brick_shape

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 14)
        self.car_shape.color = THECOLORS["gray"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        # screen.blit(car, (x, 510-y))
        #self.car_body_prev_angle = self.car_body.angle
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    

    def frame_step(self, action):
        global prev_dist, goal, car
        # print(action)
        reward = 0
        if action == 0:  # Turn left.
            self.car_body.angle -= 0.1
        elif action == 1:  # Turn right.
            self.car_body.angle += 0.1
        self.car_body.angle = self.car_body.angle%(2*math.pi)
        # Move obstacles.
        #if self.num_steps % 100 == 0:
        #    self.move_obstacles()
        #print(self.car_body.angle)
        #print(self.car_body.angle)
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        # print()
        if not goal:
            self.car_body.velocity = 10* driving_direction
        else:
            self.car_body.velocity = 0*driving_direction
        # Update the screen and stuff.	
        screen.fill(THECOLORS["gray"])
        self.create_buff_debuff((0, 255, 0), 400, 51, 54, 48)# buff zone
        # self.create_buff_debuff("green", 400, 51, 54, 48)# buff zone
        # self.create_debuff( 190, 193.5, 54, 48)#debuff zone
        self.create_buff_debuff((255, 255, 0), 50, 336, 54, 48)#supply zone
        self.create_buff_debuff((0, 0, 255), 400, 510-51, 54, 48)#supply zone
        # self.create_debuff( 810-190, 510-193.5, 54, 48)#debuff
        self.create_buff_debuff((0, 0, 255), 810-50, 510-336, 54, 48)#enemy buff
        screen.blit(blue, (0, 0))
        screen.blit(blue, (0, 410))
        screen.blit(red, (710, 0))
        screen.blit(red, (710, 410))
        
        #draw(screen, self.space)
        # pygame.display.update()
        self.space.debug_draw(self.drawoptions)
        self.space.step(1./10)
        # if draw_screen:
        #     pygame.display.flip()
        #clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        # car=pygame.transform.rotate(car, self.car_body.angle*360/(2*math.pi))
        screen.blit(car, (x-50, y-50))
        # print(x, y)
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        #print(readings)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        pygame.draw.line(screen, (255, 255, 255), (self.goal[0],510 - self.goal[1]) , (x, y))
        pygame.display.update()
        joining_line = Vec2d(self.goal[0]-x, 510-self.goal[1]-y)
        j_angle = joining_line.angle
        j_angle = j_angle*int(j_angle>=0) + (j_angle+2*math.pi)*int(j_angle<0) 
        dist = math.sqrt((self.goal[0]-x)**2+(510-self.goal[1]-y)**2)
        normalized_readings.append(dist)
        #normalized_readings.append(self.goal[1]-y)
        normalized_readings.append(j_angle - self.car_body.angle)
        state = np.array([normalized_readings]) 
        # Set the reward.
        # Car crashed when any reading == 1
        
        # print("reading",readings)
        
        reward_list = []
        temp = math.fabs(j_angle - self.car_body.angle)
        temp = temp*(temp <=math.pi) + (2*math.pi - temp)*(temp>math.pi)
        
        obs=screen.get_at((int(x), int(y-5)))
        # print(obs)
        # obs=(255, 255, 255, 255)
        # print(obs)
        # print(THECOLORS["black"])
        
        if (dist<prev_dist):
            reward=reward+ 400
            if action==2:
                reward=reward+50
        elif(dist>=prev_dist):
            reward=reward-500

        if temp < 0.4:
            reward_list.append("angle is good")
            
            # print("goood angle")
            reward += 300
        else :
            reward_list.append("angle is not good")
            # print("bad angle")
            reward -= 200
        # &(y<=(51+44))
        if self.car_at_buff(obs):
            self.time = 0
            reward = reward+10000
            # self.car_body.position= 100, 100
            print(10000)
            goal=True
            prev_dist = math.sqrt((self.goal[0]-x)**2+(510-self.goal[1]-y)**2)
            self.init_heuristic = Vec2d(self.goal[0]-x, 510-self.goal[1]-y).get_length()
            joining_line = Vec2d(self.goal[0]-x, 510-self.goal[1] - y)
            reward_list.append("goal reached")
            
            print(x, y)
            while True:
                pass
            
        elif ((x<=447) and (x>=(400-47)) and (y>=(510-(51+44)))):
            reward += 4000
            print(4000)
        elif ((x<=497) and (x>=(400-97)) and (y>=(510-(51+44)))):
            reward += 3000
            print(3000)
        elif self.car_at_debuff(obs):
            reward -= 10000
            print(-10000)
            reward_list.append("debuff")
            print(x, y)
            # print("debuff")
        elif (x<=(190+47) and (x>=(190-47)) and (y>=(510-(193+44))) and (y<=(510-(193-44)))):
            reward -= 1000
            print(-1000)
        elif (x<=(810-190+47) and (x>=(810-190-47)) and (y>=(51-44)) and (y<=(51+44))):
            reward -= 1000
            print(-1000)
        
        elif(obs==(0, 0, 255, 255)):
            reward=reward-10000
            print(-10000)
            # print("enemy buff")
            reward_list.append("enemy buff")
        elif(obs==(255, 255, 0, 255)):
            reward=reward-10000
            print(-10000)
            # print("debuff")
            reward_list.append("debuff")
        
        prev_dist=dist
        
        if self.sum_readings(readings)<160:
            reward=reward-80
        
        if self.car_is_crashed(readings):
            self.time -= 5
            self.crashed = True
            reward =  reward-500 + self.time
            self.recover_from_crash(driving_direction) 
            reward_list.append("crash")
            print("crash")
            
            
            
        elif joining_line.get_length() > 270*f2:
            self.time = self.time - 5
            # reward = reward + self.time
            # Higher readings are better, so return the sum.
            # reward = reward +(-10 + int(self.sum_readings(readings) / 10))
            #print("hello",-5 + int(self.sum_readings(readings) / 10))
            reward = reward + (self.prev_goal_distance - joining_line.get_length())*LENGTH_COEFF
            #print("bye",(self.prev_goal_distance - joining_line.get_length())*LENGTH_COEFF)
            #print(self.prev_goal_distance)
            #print(joining_line.get_length())
            reward_list.append("goal not reached")
            reward_list.append(joining_line.get_length())
            if(self.time < -50000):
                self.time = 0
                self.car_body.position= 50, 50
                reward=reward-20000
                prev_dist = math.sqrt((self.goal[0]-x)**2+(510-self.goal[1]-y)**2)


        self.prev_goal_distance = joining_line.get_length()
        

        self.num_steps += 1
        #print(self.num_steps)
        reward_list.append(reward)
        # print(self.sum_readings(readings)," ", reward)
        
        #print(reward_list)
        #print(x,y,self.goal[0],self.goal[1])
        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            #print("hello")
            speed = random.randint(1, 2)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 30)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -5 * driving_direction
            self.crashed = False
            for i in range(10):
                #self.car_body.angle += .05  # Turn a little.
                screen.fill(THECOLORS["grey"])  # Red is scary!
                #draw(screen, self.space)
                self.space.step(1./10)
                #if draw_screen:
                #    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        arms = []
        # print(angle)
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        for i in range(16):
            arms.append(self.make_sonar_arm(x,y))
            readings.append(self.get_arm_distance(arms[i],x,y,angle,(math.pi/8)*i, i))
        # Make our arms.
        
        
        #arm_left = self.make_sonar_arm(x, y)
        #arm_middle = arm_left
        #arm_right = arm_left

        # Rotate them and get readings.
        #readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        #readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        #readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset, p):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at((int(rotated_p[0]), int(510-rotated_p[1])))
                if self.get_track_or_not(obs) != 1:
                    return i

            if show_sensors:
                if(p==0):
                    pygame.draw.circle(screen, (255, 0, 0), (rotated_p[0], 510-rotated_p[1]), 2)
                else:
                    pygame.draw.circle(screen, (255, 255, 255), (rotated_p[0], 510-rotated_p[1]), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == (0, 0, 0, 255):
            return 0
        else:
            return 1
    def car_crashed(self, space, arbiter):
        if arbiter.is_first_contact:
            for contact in arbiter.contacts:
                self.crashed = True

    def car_at_buff(self, color):
        if color == (0, 255, 0, 255):
            return 1
        else:
            return 0
        
    def car_at_debuff(self, color):
        if color == (255, 0, 0, 255):
            return 1
        return 0
    
    def reset_goal(self, x, y):
        # self.goal = (int(random.random()*29031)%8100*f2, int(random.random()*10093)%5100*f2)
        # while not (self.goal[0] > x1+90 or self.goal[0] < x4-90 or self.goal[1] > y1+90 or self.goal[1] < y3-90 or screen.get_at((int(self.goal[0]), int(510-self.goal[1]))) != (0, 0, 0, 255)):
        #     self.goal = (int(random.random()*29031)%8100*f2, int(random.random()*10093)%5100*f2)
        x=100
        y=100
        return(x, y)
        
if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step(np.random.randint(3))
