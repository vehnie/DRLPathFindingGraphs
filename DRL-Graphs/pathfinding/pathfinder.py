import networkx as nx
import queue
import time
import numpy as np
import tensorflow as tf

class PathFinder:
    def __init__(self, graph, start_node, goal_node):
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.training_queue = queue.Queue()
        self.stop_training = False
        
    def heuristic(self, node):
        """Manhattan distance heuristic for A*."""
        # For now, use a simple distance of 1 since we don't have coordinates
        return 1
        
    def run_astar(self):
        """A* pathfinding algorithm."""
        frontier = []  # Priority queue
        frontier.append((0, self.start_node))
        came_from = {}
        cost_so_far = {}
        came_from[self.start_node] = None
        cost_so_far[self.start_node] = 0
        current_path_edges = set()
        explored_edges = set()
        final_path = None
        
        while frontier and not self.stop_training:
            current = frontier.pop(0)[1]
            
            # Update current path visualization
            current_path_edges = set()
            temp_node = current
            while temp_node in came_from and came_from[temp_node] is not None:
                current_path_edges.add((came_from[temp_node], temp_node))
                temp_node = came_from[temp_node]
            
            # Send update through queue
            self.training_queue.put({
                'current_episode_edges': current_path_edges.copy(),
                'past_episode_edges': explored_edges.copy(),
                'score': -cost_so_far[current],
                'final_path': None,
                'training_finished': False
            })
            time.sleep(0.1)
            
            if current == self.goal_node:
                # Reconstruct final path
                path = []
                temp_node = current
                while temp_node in came_from and came_from[temp_node] is not None:
                    path.append((came_from[temp_node], temp_node))
                    temp_node = came_from[temp_node]
                path.reverse()
                final_path = path
                break
                
            for next_node in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + 1  # Assuming uniform edge cost of 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node)
                    # Insert maintaining sorted order
                    i = 0
                    while i < len(frontier) and frontier[i][0] < priority:
                        i += 1
                    frontier.insert(i, (priority, next_node))
                    came_from[next_node] = current
                    explored_edges.add((current, next_node))
        
        # Send final update
        final_score = -cost_so_far[self.goal_node] if self.goal_node in cost_so_far else float('-inf')
        self.training_queue.put({
            'current_episode_edges': current_path_edges.copy(),
            'past_episode_edges': explored_edges.copy(),
            'score': final_score,
            'final_path': final_path,
            'training_finished': True
        })
        time.sleep(0.1)  # Ensure the final update is processed
        
        return final_path

    def run_dijkstra(self):
        """Dijkstra's pathfinding algorithm."""
        frontier = []  # Priority queue
        frontier.append((0, self.start_node))
        came_from = {}
        cost_so_far = {}
        came_from[self.start_node] = None
        cost_so_far[self.start_node] = 0
        current_path_edges = set()
        explored_edges = set()
        final_path = None
        
        while frontier and not self.stop_training:
            current = frontier.pop(0)[1]
            
            # Update current path visualization
            current_path_edges = set()
            temp_node = current
            while temp_node in came_from and came_from[temp_node] is not None:
                current_path_edges.add((came_from[temp_node], temp_node))
                temp_node = came_from[temp_node]
            
            # Send update through queue
            self.training_queue.put({
                'current_episode_edges': current_path_edges.copy(),
                'past_episode_edges': explored_edges.copy(),
                'score': -cost_so_far[current],
                'final_path': None,
                'training_finished': False
            })
            time.sleep(0.1)
            
            if current == self.goal_node:
                # Reconstruct final path
                path = []
                temp_node = current
                while temp_node in came_from and came_from[temp_node] is not None:
                    path.append((came_from[temp_node], temp_node))
                    temp_node = came_from[temp_node]
                path.reverse()
                final_path = path
                break
                
            for next_node in self.graph.neighbors(current):
                new_cost = cost_so_far[current] + 1  # Assuming uniform edge cost of 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # Insert maintaining sorted order by cost
                    i = 0
                    while i < len(frontier) and frontier[i][0] < new_cost:
                        i += 1
                    frontier.insert(i, (new_cost, next_node))
                    came_from[next_node] = current
                    explored_edges.add((current, next_node))
        
        # Send final update
        final_score = -cost_so_far[self.goal_node] if self.goal_node in cost_so_far else float('-inf')
        self.training_queue.put({
            'current_episode_edges': current_path_edges.copy(),
            'past_episode_edges': explored_edges.copy(),
            'score': final_score,
            'final_path': final_path,
            'training_finished': True
        })
        time.sleep(0.1)  # Ensure the final update is processed
        
        return final_path

class DeepQAgent:
    def __init__(self, state_dim, action_dim, learning_rate, discount_factor, memory_size, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.model = self._build_model()

    def _build_model(self):
        # Build the Deep Q-Network model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='mse')
        return model

    def act(self, state):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < 0.1:
            return np.random.randint(0, self.action_dim)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self):
        # Train on batch of experiences
        if len(self.memory) < self.batch_size:
            return None
        batch = np.random.choice(self.memory, self.batch_size, replace=False)
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])
        targets = rewards + self.discount_factor * np.max(self.model.predict(next_states), axis=1) * (1 - dones)
        self.model.fit(states, targets, epochs=1, verbose=0)
        return self.model.loss

class DeepQLearningPathFinder(PathFinder):
    def __init__(self, graph, start_node, goal_node):
        super().__init__(graph, start_node, goal_node)
        self.agent = DeepQAgent(len(graph.nodes()), len(graph.nodes()), 
                              learning_rate=0.001, 
                              discount_factor=0.99,
                              memory_size=10000,
                              batch_size=32)
        self.best_path = []
        self.best_reward = float('-inf')
        self.past_edges = set()

    def train_agent(self, num_episodes):
        """Train the Deep Q-Learning agent."""
        for episode in range(num_episodes):
            if self.stop_training:
                break

            current_state = self.start_node
            total_reward = 0
            path = []
            self.current_edges = set()
            done = False

            while not done and not self.stop_training:
                # Clear current edges from previous step
                self.current_edges.clear()
                
                # Choose action using epsilon-greedy policy
                action = self.agent.act(self.state_to_tensor(current_state))

                # Take action and observe next state and reward
                if action in self.graph.neighbors(current_state):
                    next_state = action
                    if next_state == self.goal_node:
                        reward = 100
                        done = True
                    else:
                        reward = -1
                else:
                    next_state = current_state
                    reward = -10

                # Update current path and edges
                if current_state != next_state:
                    edge = (min(current_state, next_state), max(current_state, next_state))
                    path.append(edge)
                    self.current_edges.add(edge)
                    self.past_edges.add(edge)

                # Store experience in memory
                self.agent.remember(self.state_to_tensor(current_state),
                                 action,
                                 reward,
                                 self.state_to_tensor(next_state),
                                 done)

                # Update total reward and current state
                total_reward += reward
                current_state = next_state

                # Train on batch of experiences
                self.agent.replay()

                # Send immediate update for visualization
                self.training_queue.put({
                    'episode': episode + 1,
                    'total_reward': total_reward,
                    'best_reward': self.best_reward,
                    'current_edges': self.current_edges.copy(),
                    'past_edges': self.past_edges.copy(),
                    'best_path': None,
                    'training_finished': False
                })

                if done:
                    # Update best path if this episode was better
                    if total_reward > self.best_reward:
                        self.best_reward = total_reward
                        self.best_path = path.copy()

        # Send final update with best path highlighted
        self.training_queue.put({
            'episode': num_episodes,
            'total_reward': total_reward,
            'best_reward': self.best_reward,
            'current_edges': set(),
            'past_edges': self.past_edges.copy(),
            'best_path': self.best_path,
            'training_finished': True
        })

    def state_to_tensor(self, node):
        """Convert node state to tensor format for DQL."""
        state = np.zeros(len(self.graph.nodes()))
        state[node] = 1
        return state
