import gymnasium as gym
from gymnasium import spaces
from graphfactory.creationgraph import CreationGraph
from deep_q_learning import DeepQAgent
import random

#teste de ambiente, depois tenho de separar em dois arquivos diferentes

class GraphNavigationEnv(gym.Env):
    def __init__(self, graph, start_node, target_node):
        super(GraphNavigationEnv, self).__init__()
        self.graph = graph
        self.start_node = start_node
        self.current_node = start_node
        self.target_node = target_node
        self.visited_nodes = set([start_node])

        self.action_space = spaces.Discrete(len(graph.nodes))
        self.observation_space = spaces.Discrete(len(graph.nodes))

    def reset(self):
        self.current_node = self.start_node
        self.visited_nodes = set([self.start_node])
        return self.current_node

    def step(self, action):
        if action in self.graph.neighbors(self.current_node):
            self.current_node = action
            self.visited_nodes.add(action)

            # Calcular recompensa
            reward = -1  # penalidade base por cada movimento
            if action == self.target_node:
                reward = 10  # recompensa para alcançar o objetivo

            # Verificar se o agente visitou todos os nós
            done = action == self.target_node or len(self.visited_nodes) == len(self.graph.nodes)
        else:
            # Penalidade por tentar um movimento inválido
            reward = -2
            done = False

        info = {"visited_nodes": self.visited_nodes}
        return self.current_node, reward, done, info

    def render(self):
        print(f"Current Node: {self.current_node}, Visited Nodes: {self.visited_nodes}")


if __name__ == "__main__":
    graph_creator = CreationGraph(predefined=True, digraph=True, size='small')
    G = graph_creator.get_graph()
    START = random.randint(0,G.order()-1)
    END = random.randint(0,G.order()-1)

    # Makes sure START NODE is different from END NODE
    while START == END:
        END = random.randint(0,G.order()-1)

    env = GraphNavigationEnv(G, start_node=START, target_node=END)

    agent = DeepQAgent(state_size=1, action_size=len(G.nodes))

    episodes = 100
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act([state])
            next_state, reward, done, info = env.step(action)
            agent.remember([state], action, reward, [next_state], done)
            agent.replay()
            state = next_state
            total_reward += reward

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    print("Training completed!")
