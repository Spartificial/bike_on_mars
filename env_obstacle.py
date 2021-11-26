import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
from decimal import Decimal

### Bicycle and its environment

class CycleBalancingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Out cycle has only 2 action spaces i.e. Torque of the wheels and the position of the handlebar
        self.action_space = gym.spaces.box.Box(
            low=-1 * np.ones(1, dtype=np.float32),
            high=1 * np.ones(1, dtype=np.float32))

        # Obervation space
        self.observation_space = gym.spaces.box.Box(
            low=-1 * np.ones(24, dtype=np.float32),
            high=1 * np.ones(24, dtype=np.float32))

        self.np_random, _ = gym.utils.seeding.np_random()

        if not p.isConnected():
            self.client = p.connect(p.GUI)  # Physics + Visual
            # self.client = p.connect(p.DIRECT) # Only Physics, no visualization. For faster training
        else:
            self.client = 1

        self.n_target = 200  # Number of obstacles
        self.min_target_dist = 5  # Minimum distance of obstacles from bike
        self.target_span = 100  # Maximum distance of obstacles from bike
        self.sphere_dist = 1.5
        # self.pole = []

        p.resetSimulation(self.client)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)  # Loading the Plane Surface
        self.bike = 0  # Index of the Bike
        self.angle_span = 20  # The Angle between two consecutive rays
        self.n_episodes = 0  # Count of the number of episodes
        self.rays_distance = 30  # The max length of the rays
        self.z_balance = -0.25  # Offset between the Bike's center of gravity and the height from which rays are passed

        self.make_obstacles()  # create the obstacles
        self.reset()  # Reseting the environment

    # Helper (Debugger) function to show the distance traveled by rays in all direction
    def show_img(self):

        self.img = np.zeros((800, 800, 3), dtype='float32')
        shift = 400
        multiply = 400
        ls = p.getBasePositionAndOrientation(self.bike)
        bike_x = ls[0][0]
        bike_y = ls[0][1]
        handlebar_rotation = p.getEulerFromQuaternion(p.getLinkState(self.bike, 0)[1])[2]
        mini = 1000
        for deg in range(1, 361, 1):
            mini = min(mini, self.dist[deg - 1])
            if deg % self.angle_span == 0:
                rad = Decimal(Decimal(deg * np.pi / 180 + handlebar_rotation) % Decimal(2 * np.pi) + Decimal(
                    2 * np.pi)) % Decimal(2 * np.pi)
                rad = float(rad)
                start = (int(shift + bike_x + self.sphere_dist * np.cos(rad)),
                         int(shift + bike_y + self.sphere_dist * np.sin(rad)))
                end = (int(shift + bike_x + mini * multiply * np.cos(rad)),
                       int(shift + bike_y + mini * multiply * np.sin(rad)))
                cv2.ellipse(self.img, start, (int(mini * multiply), int(mini * multiply)), 0,
                            (rad * 180 / np.pi) - self.angle_span, (rad * 180 / np.pi), (0, 0, 255), -1)
                mini = 1000

        cv2.imshow('img', cv2.rotate(cv2.transpose(self.img), cv2.ROTATE_180))
        # cv2.imshow('img', self.img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

    def apply_action(self, action):
        p.setJointMotorControl2(self.bike, 0, p.POSITION_CONTROL, targetPosition=action[0],
                                maxVelocity=5)  # Apply Position control to Handlebar

    def apply_torque_wheels(self):
        p.setJointMotorControl2(self.bike, 1, p.TORQUE_CONTROL, force=(2.5 + 0) * 10000)  # Apply Toruqe to Back Wheel
        p.setJointMotorControl2(self.bike, 2, p.TORQUE_CONTROL, force=(2.5 + 0) * 10000)  # Apply Toruqe to Front Wheel

    def gyroscope_torque(self):
        ls = p.getBasePositionAndOrientation(self.bike)  # ls[0]=Postion of cycle, ls[1] = Orientation of cycle
        val = p.getEulerFromQuaternion(ls[1])[0] - 1.57  # Calculating inclination of cycle from vertical
        p.applyExternalTorque(self.bike, -1, [-1000000 * val, 0, 0], flags=p.WORLD_FRAME)

    def add_pos_orien(self):
        ls = p.getBasePositionAndOrientation(self.bike)  # Calculating the postion and orientation of the Bike
        # ls[0]=Postion of cycle, ls[1] = Orientation of cycle
        self.obs += ls[0]  # Adding postion
        self.obs += p.getEulerFromQuaternion(ls[1])  # Adding orientation

    def add_dist_by_rays(self):
        ls = p.getBasePositionAndOrientation(self.bike)  # Calculating the postion and orientation of the Bike

        z = ls[0][2] + self.z_balance  # Height of the bike above the ground (Used by rays)

        self.bike_x = ls[0][0]  # X-position of the Bike
        self.bike_y = ls[0][1]  # Y-position of the Bike

        # Calculating the positon and length of the rays (i.e. start and end points of the rays)
        bike_x = ls[0][0]
        bike_y = ls[0][1]
        reward_2 = 0
        ray_from = []
        ray_to = []
        handlebar_rotation = p.getEulerFromQuaternion(p.getLinkState(self.bike, 0)[1])[2]
        for deg in range(1, 361, 1):
            rad = Decimal(
                Decimal(deg * np.pi / 180 + handlebar_rotation) % Decimal(2 * np.pi) + Decimal(2 * np.pi)) % Decimal(
                2 * np.pi)
            rad = float(rad)
            for i in np.arange(0, 3, 1):
                for j in np.arange(-4, 1, 0.5):
                    ray_from.append(
                        (bike_x + self.sphere_dist * np.cos(rad), bike_y + self.sphere_dist * np.sin(rad), z + i))
                    ray_to.append(
                        (bike_x + self.rays_distance * np.cos(rad), bike_y + self.rays_distance * np.sin(rad), z + j))

        # Adding the observation of the rays (i.e whether the rays collided with any object or not)
        rays = p.rayTestBatch(ray_from, ray_to)
        mini = 1000
        self.dist = []
        mini = 1000
        cnt = 0
        for deg in range(1, 361, 1):
            dist = 1
            for i in np.arange(0, 3, 1):
                for j in np.arange(-4, 1, 0.5):
                    tmp = rays[cnt]
                    cnt += 1
                    if tmp[0] != self.plane: dist = min(dist, tmp[2])
            if dist < mini:
                mini = dist
            self.dist.append(dist)
            if deg % self.angle_span == 0:
                self.obs.append(mini)
                mini = 1000

    def get_collision(self):
        ls = p.getContactPoints(self.bike)
        reward = 0
        done = False
        for i in range(len(ls)):
            if ls[i][2] != 0:
                reward = -100
                done = True
        return reward, done

    def prevent_rotating_in_a_cycle(self):
        ls = p.getBasePositionAndOrientation(self.bike)  # Calculating the postion and orientation of the Bike
        value = p.getEulerFromQuaternion(p.getLinkState(self.bike, 0)[1])[2] - p.getEulerFromQuaternion(ls[1])[2]
        if value < -1: value += 2 * np.pi
        if value > 1: value -= 2 * np.pi
        # print(value)
        if value < -0.5:
            self.left += 0.1
            self.right = 0
        elif value > 0.5:
            self.left = 0
            self.right += 0.1
        else:
            self.left = 0
            self.right = 0

        self.neg_reward = 0
        if self.left > 10 or self.right > 10:
            print("Slow!!!")
            self.neg_reward = -100
            self.done = True

    def target_dist_achieved(self):
        reward_1 = 0
        dist_2 = np.sqrt((self.bike_x) ** 2 + abs(self.bike_y) ** 2)
        if dist_2 > self.target_dist:
            reward_1 = 100 + self.target_reward
            self.target_dist += 10
            self.target_reward = min(500, self.target_reward * 2)

        self.completed = 0
        if dist_2 > self.target_span:
            self.completed = 1
            self.done = True
            print("DONE!")
            self.make_obstacles()
        return reward_1

    def get_reward(self, reward_1, reward_2):
        dist_2 = np.sqrt((self.bike_x) ** 2 + abs(self.bike_y) ** 2)
        if self.time % 10 == 0 and dist_2 > self.distance:
            self.distance = dist_2
        return reward_1 + reward_2 + max(-10, (
                    dist_2 - self.distance)) + self.time / 1000 + self.neg_reward - self.left - self.right

    def get_obs(self):
        # For storing all the observations which will be used by the agent to decide the next action
        self.obs = []

        # Add bike's position, orientation to the observation space
        self.add_pos_orien()

        # Add info from rays to the observation space (i.e whether rays collided with ant object or not)
        self.add_dist_by_rays()

        self.obs = np.array(self.obs, dtype=np.float32)
        self.obs[0] /= self.target_span
        self.obs[1] /= self.target_span

    # Step function
    def step(self, action):

        self.apply_action(action)

        for i in range(3):
            self.apply_torque_wheels()  # Apply Torque to the wheels (i.e increase velocity)
            self.gyroscope_torque()  # Balancing by applying Torque (In real world, this will be done by gyroscope)
            p.stepSimulation()

        self.get_obs()  # Get the observation

        # Terminate the episode if the Bike collided with any obstacle
        reward_2, self.done = self.get_collision()

        # Adding 1 to the time for which the current episode has been running
        self.time += 1

        # Terminate the episode if the Bike keeps rotating in a circle
        self.prevent_rotating_in_a_cycle()

        # Terminating the episode if the cycle is more than "target_span" distance away from the origin
        reward_1 = self.target_dist_achieved()

        # Calculating the total reward
        reward = self.get_reward(reward_1, reward_2)

        return self.obs, reward / 100, self.done, dict()

    def load_bike(self):
        # Remove the Bike if already loaded
        if self.bike != 0:
            p.removeBody(self.bike)

        # Loading the Bike
        self.bike_x = 0  # random.randint(-5, 5) # X position of the Bike
        self.bike_y = 0  # random.randint(-5, 5) # Y position of the Bike
        # path = os.getcwd()
        self.bike = p.loadURDF('bike_2.urdf.xml', [self.bike_x, self.bike_y, 0],
                               p.getQuaternionFromEuler([0, 0, random.random() * 2 * np.pi]), useFixedBase=False)

    def add_dynamics(self):
        # Adding friction and other dynamics
        p.changeDynamics(self.plane, -1, lateralFriction=5, angularDamping=1)
        p.changeDynamics(self.bike, 1, mass=100)
        p.changeDynamics(self.bike, -1, lateralFriction=5, angularDamping=1)

        p.setGravity(0, 0, -250)  # Setting the gravity

    def reset(self):

        self.n_episodes += 1  # Increase the number of episodes by 1

        # Change obstacles's position after every 20 episodes for robust training
        if self.n_episodes == 20:
            self.make_obstacles()
            self.n_episodes = 0

        self.load_bike()

        for i in range(50):
            p.stepSimulation()

        self.add_dynamics()

        self.done = False
        self.time = 0
        self.distance = np.sqrt((self.bike_x) ** 2 + abs(self.bike_y) ** 2)
        self.neg_reward = 0

        self.get_obs()  # Get the observation

        # Initialize variables
        self.cnt = 0
        self.left = 0
        self.right = 0
        self.target_dist = 10
        self.target_reward = 32
        self.completed = 0

        return self.obs

    # Function to create the obstacles
    def make_obstacles(self):

        p.resetSimulation(self.client)
        p.setRealTimeSimulation(0)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the obstacles
        mul = 100  # Factor to increase or decrease the size of the obstacles
        height = 2
        visualShift = [0, 0, 0]
        shift = [0, 0, 0]
        meshScale = [0.1 * mul, 0.1 * mul, 0.1 * mul * height]
        # path = 'C:/Users/User/Documents/GitHub/bullet3/examples/pybullet/gym/pybullet_data/'
        groundColId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                             fileName="terrain.obj",
                                             collisionFramePosition=shift,
                                             meshScale=meshScale,
                                             flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        groundVisID = p.createVisualShape(shapeType=p.GEOM_MESH,
                                          fileName="terrain.obj",
                                          rgbaColor=[0.7, 0.3, 0.1, 1],
                                          specularColor=[0.4, .4, 0],
                                          visualFramePosition=visualShift,
                                          meshScale=meshScale)
        self.plane = p.createMultiBody(baseMass=0,
                                       baseInertialFramePosition=[0, 0, 0],
                                       baseCollisionShapeIndex=groundColId,
                                       baseVisualShapeIndex=groundVisID,
                                       basePosition=[0, 0, 0],
                                       useMaximalCoordinates=True)

        self.bike = 0
        self.bike_x = 0
        self.bike_y = 0
        self.pole = []
        # Load the Target
        for i in range(self.n_target):
            target_x = self.bike_x
            target_y = self.bike_y
            while (np.sqrt((self.bike_x - target_x) ** 2 + (self.bike_y - target_y) ** 2)) < self.min_target_dist:
                target_x = random.randint(int(self.bike_x) - self.target_span, int(self.bike_x) + self.target_span)
                target_y = random.randint(int(self.bike_y) - self.target_span, int(self.bike_y) + self.target_span)
            self.pole.append(
                p.loadURDF("cube.urdf", [target_x, target_y, 4], [0, 0, 0, 1], useFixedBase=True, globalScaling=1.0))
            p.changeDynamics(self.pole[i], -1, mass=1000)

    # Render the output Visual
    def render(self, mode='human'):
        distance = 5
        yaw = 0
        humanPos, humanOrn = p.getBasePositionAndOrientation(self.bike)
        humanBaseVel = p.getBaseVelocity(self.bike)
        # print("frame",frame, "humanPos=",humanPos, "humanVel=",humanBaseVel)
        camInfo = p.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [0.95 * curTargetPos[0] + 0.05 * humanPos[0], 0.95 * curTargetPos[1] + 0.05 * humanPos[1],
                     curTargetPos[2]]

        p.resetDebugVisualizerCamera(distance, 270, pitch, targetPos)

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]