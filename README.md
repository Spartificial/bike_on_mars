
# Self-balancing bicycle
An Open Source model of a bicycle agent that self-balances itself and navigates to a certain target location in a Martian Surface Environment.  

## Contents
- [Background](#background)
- [Mars Environment](#mars-environment-overview)
- [Types of Tasks](#types-of-tasks)
  - [Obstacles Avoidance](#obstacle-avoidance)
  - [Target Following](#target-following)
- [Solution](#solution)

## Background

The hypothetical colonization of Mars has received interest from public space agencies and private corporations, and has received extensive treatment in science fiction writing, film, and art. Reasons for colonizing Mars include curiosity, the potential for humans to provide more in-depth observational research than unmanned rovers, economic interest in its resources, and the possibility that the settlement of other planets could decrease the likelihood of human extinction. Thus for this purpose the first step is the explporation of the martian surface and many space agencies across the world have sent their rovers to mars for this purpose of exploration. Nasa's Perseverance rover is currently one of the best rovers to have been sent to Mars. But by Earth vehicle standards, the Perseverance rover is slow. By Martian vehicle standards, however, Perseverance is a standout performer. The rover has a top speed on flat, hard ground of 4.2-centimeters per second, or 152 meters per hour. This is a little less than 0.1-miles per hour. For comparison, a 3 mile-per-hour walking pace is 134 centimeters per second, or 4,828 meters per hour.

So our objective is to create a bike which is able to explore the Martian Surface faster and remains energy efficient.

##  Mars Environment Overview

1. SMALL PLANET: If the Sun were as tall as a typical front door, Earth would be the size of a dime, and Mars would be about as big as an aspirin tablet.
2. LONGER DAYS: One day on Mars takes a little over 24 hours. Mars makes a complete orbit around the Sun (a year in Martian time) in 687 Earth days.
3. RUGGED TERRAIN: Mars is a rocky planet. Its solid surface has been altered by volcanoes, impacts, winds, crustal movement and chemical reactions.
4. ATMOSPHERE: Mars has a thin atmosphere made up mostly of carbon dioxide (CO2), argon (Ar), nitrogen (N2), and a small amount of oxygen and water vapor.
5. MANY MISSIONS: Several missions have visited this planet, from flybys and orbiters to rovers on the surface.The first true Mars mission success was the Mariner 4 flyby in 1965.
6. TOUGH PLACE FOR LIFE: At this time, Mars' surface cannot support life as we know it. Current missions are determining Mars' past and future potential for life.
7. RUSTY PLANET: Mars is known as the Red Planet because iron minerals in the Martian soil oxidize, or rust, causing the soil and atmosphere to look red.

## Types of Tasks
- #### Obstacle Avoidance
        In this task our goal is to explore the martian surface while avoiding the obstacles.
- #### Target Following
        In this task our goal is to reach a given target location on Martian Surface autonomously.
        
## Solution
For both of the above task we used a Reinforcement Learning (RL) approach. 

### Algorithm
Deep Deterministic Policy Gradient (DDPG) is a reinforcement learning technique that combines both Q-learning and Policy gradients. DDPG being an actor-critic technique consists of two models: Actor and Critic. The actor is a policy network that takes the state as input and outputs the exact action (continuous), instead of a probability distribution over actions. The critic is a Q-value network that takes in state and action as input and outputs the Q-value. DDPG is an “off”-policy method. DDPG is used in the continuous action setting and the “deterministic” in DDPG refers to the fact that the actor computes the action directly instead of a probability distribution over actions.
DDPG is used in a continuous action setting and is an improvement over the vanilla actor-critic.

###Procedure:
1. Firstly we create an OpenAI gym environment where we can control our agent (Bike) in an environment (Martian Surface).
2. Now we setup a reward function (R) and the goal of our agent will be to maximize this reward function.
3. Now we will train our agent to maximize this reward function using Reinforcement Learning algorithms (DDPG algorithm) by using experience of self-play.
4. PLoting the cummulative rewaard to see if the agent's trainging is converging or not.
5. Saving and testing the model.

### Components of RL
- Agent: Bike
- Environment: Mars surface and obstacles (stones)
- Action: The Bike's Handlebar position (i.e the angle at which the handlebar should be moved)
- Observation: 
    1. Postion and Orientation of the Bike.
    2. Postion of the Target location.
    3. The distance traveled by rays before colliding with an obstacle.
- Reward:
    1. Positive reward if the Bike reaches the target location.
    2. Negative reward if the Bike collides with an obstacle.
    3. Negative reward if the Bike gets too far away from the target location.
    4. Negative reward if the Bike keeps rotating in a circular path.
