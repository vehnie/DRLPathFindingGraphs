import gymnasium as gym
from gymnasium import spaces
from graphfactory.creationgraph import CreationGraph
import random


class GraphNavigationEnv(gym.Env):
    def __init__(self, graph, start_node, target_node=None):
        super(GraphNavigationEnv, self).__init__()
        self.graph = graph
        self.start_node = start_node
        self.current_node = start_node
        self.target_node = target_node
        self.visited_nodes = set([start_node])
        
        # Definir o limite de movimentos como o triplo do número de vértices
        self.max_steps = 3 * len(graph.nodes)
        self.steps_taken = 0

        self.action_space = spaces.Discrete(len(graph.nodes))
        self.observation_space = spaces.Discrete(len(graph.nodes))

    def reset(self):
        self.current_node = self.start_node
        self.visited_nodes = set([self.start_node])
        self.steps_taken = 0 
        return self.current_node

    def step(self, action):
        # Verificar se o agente atingiu o limite de movimentos
        if self.steps_taken >= self.max_steps:
            done = True
            reward = 0 
            info = {"visited_nodes": self.visited_nodes}
            return self.current_node, reward, done, info
        
        if action in self.graph.neighbors(self.current_node):
            self.current_node = action
            self.visited_nodes.add(action)
            self.steps_taken += 1  

            # Calcular recompensa
            reward = -1  # Penalidade por cada movimento
            if action == self.target_node:
                reward = 10  # Recompensa para alcançar o objetivo

            # Verificar se o agente visitou todos os nós ou atingiu o objetivo
            done = action == self.target_node or len(self.visited_nodes) == len(self.graph.nodes)
        else:
            # Penalidade por tentar um movimento inválido
            reward = -2
            done = False

        info = {"visited_nodes": self.visited_nodes}
        return self.current_node, reward, done, info

    def render(self, mode='human'):
        print(f"Current Node: {self.current_node}, Visited Nodes: {self.visited_nodes}")


if __name__ == "__main__":
    graph_creator = CreationGraph()  
    G = graph_creator.get_graph(n=5, p=0.2, directed=False)
    node = random.choice(list(G.nodes))

    env = GraphNavigationEnv(G, start_node=node, target_node=5)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  
        state, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

    print("Total Reward:", total_reward)
