import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.rate = 0.1
        self.gamma = 0.9
        self.state = None
        self.q_learn = {}
        self.runs = 0
        self.valid_actions = self.env.valid_actions

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.runs += 1
        if self.runs >= 100:
            self.rate = 0
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'])

        #initialize q_learn dictionary by passing state vector and setting q = 1 for long term learning
        for action in self.valid_actions:
            if (self.state, action) not in self.q_learn:
                self.q_learn[(self.state, action)] = 1
        # TODO: Select action according to your policy

        chances = [self.q_learn[(self.state, None)], self.q_learn[(self.state, 'forward')], self.q_learn[(self.state, 'left')], self.q_learn[(self.state, 'right')]]
        chances = np.exp(chances) / np.sum(np.exp(chances), axis=0)
        exploration = max((100 - self.runs) / 100 , 0) # this line of code decides whether smartcab should choose a random action
                                                        # based on self.trips taken. As more trips are taken the cab will rule out
                                                        # chance and decide on max values from q_learning.
        #action = random.choice(self.env.valid_actions[1:]) #random action

        if random.random() < exploration:
            action = np.random.choice(self.valid_actions, p = chances)
        else:#this statement chooses max q_value and if there are multiple max values it randomly chooses one of them
            likely = max(chances)
            holder = [index for index in range(len(chances)) if likely == chances[index]]
            entry = np.random.choice(holder)
            action = self.valid_actions[entry]

        # Execute action and get reward
        reward = self.env.act(self, action)

        new_next_inputs = self.env.sense(self)
        next_state = (self.planner.next_waypoint(), new_next_inputs['light'], new_next_inputs['oncoming'], new_next_inputs['left'], new_next_inputs['right'])

        for a in self.valid_actions:
            if (next_state, a) not in self.q_learn:
                self.q_learn[(next_state, a)] = 1

        new_q_values = [self.q_learn[(next_state, a)] for a in self.valid_actions]
        max_q = max(new_q_values)

        # TODO: Learn policy based on state, action, reward
        self.q_learn[(self.state, action)] = self.rate * reward + (1 - self.rate) * self.q_learn[(self.state, action)] + self.rate * self.gamma * max_q

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline = True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay = 0, display = False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials = 100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()