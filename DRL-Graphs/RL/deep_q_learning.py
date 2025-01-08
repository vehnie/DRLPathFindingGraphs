import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import queue
import os
import pickle
from collections import deque, namedtuple
import networkx as nx
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import threading
import dash
import time

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class DeepQLearningPathFinder:
    def __init__(self, graph, start_node, goal_node, learning_rate=0.001, discount_factor=0.99, exploration_rate=1.0,
                 no_repeat_nodes=False, no_repeat_edges=False, memory_size=10000, batch_size=32, hidden_size=64,
                 target_update=10, model_path='model.pth', memory_path='memory.pkl'):
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        self.no_repeat_nodes = no_repeat_nodes
        self.no_repeat_edges = no_repeat_edges
        self.training_queue = queue.Queue()
        self.stop_training = False
        self.best_path = None
        self.best_reward = float('-inf')
        
        # DQN specific parameters
        self.memory_size = memory_size
        self.batch_size = min(batch_size, 16)
        self.memory = deque(maxlen=memory_size)
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.memory_path = os.path.join(os.path.dirname(__file__), memory_path)
        
        # Initialize state and action dimensions
        self.nodes_list = sorted(list(graph.nodes()))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes_list)}
        self.num_nodes = len(self.nodes_list)
        self.state_dim = self.num_nodes
        self.action_dim = self.num_nodes
        
        # Initialize networks
        self.device = torch.device('cpu')
        self.policy_net = DQN(self.state_dim, hidden_size, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, hidden_size, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.target_update_counter = 0
        self.target_update = target_update
        
        # Load existing model and memory if available
        self.load_model_and_memory()
        
    def load_model_and_memory(self):
        try:
            if os.path.exists(self.model_path):
                try:
                    state_dict = torch.load(self.model_path, map_location=self.device)
                    if all(self.policy_net.state_dict()[k].shape == state_dict[k].shape 
                          for k in self.policy_net.state_dict().keys()):
                        self.policy_net.load_state_dict(state_dict)
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    else:
                        if os.path.exists(self.model_path):
                            os.remove(self.model_path)
                except Exception as e:
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    print(f"Removed incompatible model file: {e}")

            if os.path.exists(self.memory_path):
                try:
                    with open(self.memory_path, 'rb') as f:
                        self.memory = pickle.load(f)
                except Exception as e:
                    if os.path.exists(self.memory_path):
                        os.remove(self.memory_path)
                    print(f"Removed incompatible memory file: {e}")
                    self.memory = deque(maxlen=self.memory_size)
        except Exception as e:
            print(f"Error in load_model_and_memory: {e}")
            self.memory = deque(maxlen=self.memory_size)

    def save_model_and_memory(self):
        try:
            torch.save(self.policy_net.state_dict(), self.model_path)
            with open(self.memory_path, 'wb') as f:
                pickle.dump(self.memory, f)
        except Exception as e:
            print(f"Error saving model or memory: {e}")

    def state_to_tensor(self, current_node, visited_nodes):
        state = np.zeros(self.num_nodes)
        state[self.node_to_idx[current_node]] = 1
        return torch.FloatTensor(state).to(self.device)

    def get_valid_actions(self, current_node, visited_nodes):
        neighbors = list(self.graph.neighbors(current_node))
        if self.no_repeat_nodes:
            neighbors = [n for n in neighbors if n not in visited_nodes]
        return neighbors if neighbors else []

    def get_action(self, state_tensor, valid_actions):
        if not valid_actions:
            return None
            
        if random.random() < self.exploration_rate:
            return random.choice(valid_actions)
            
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            valid_q_values = [(action, q_values[self.node_to_idx[action]].item()) for action in valid_actions]
            return max(valid_q_values, key=lambda x: x[1])[0]

    def get_reward(self, current_node, next_node, done):
        if done and next_node == self.goal_node:
            return 100
        elif done:
            return -50
        else:
            return -1

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*transitions))
        
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor([self.node_to_idx[a] for a in batch.action], device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            next_q_values[done_batch] = 0
            target_q_values = reward_batch + self.discount_factor * next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Log training metrics
        print(f"\rLoss: {loss.item():.4f}, Avg Q-Value: {current_q_values.mean().item():.4f}", end="")

    def train(self, num_episodes=100):
        """Train the agent using Deep Q-Learning."""
        episode_rewards = []
        self.best_reward = float('-inf')
        self.best_path = None
        
        for episode in range(num_episodes):
            if self.stop_training:
                break
                
            current_node = self.start_node
            episode_reward = 0
            current_path = [(None, self.start_node)]  # Track full path
            done = False
            
            while not done and current_node != self.goal_node:
                if self.stop_training:
                    break
                
                state = self.state_to_tensor(current_node, [node for _, node in current_path])
                valid_actions = self.get_valid_actions(current_node, [node for _, node in current_path])
                
                if not valid_actions:
                    break
                
                # Choose action using epsilon-greedy policy
                if random.random() < self.exploration_rate:
                    next_node = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = self.policy_net(state)
                    valid_q_values = [(action, q_values[self.node_to_idx[action]].item()) for action in valid_actions]
                    next_node = max(valid_q_values, key=lambda x: x[1])[0]
                
                # Take action and observe reward
                reward = self.get_reward(current_node, next_node, next_node == self.goal_node)
                episode_reward += reward
                next_state = self.state_to_tensor(next_node, [node for _, node in current_path] + [next_node])
                
                # Store transition in memory
                self.memory.append(Experience(state, next_node, reward, next_state, next_node == self.goal_node))
                
                # Move to next state
                current_node = next_node
                current_path.append((current_path[-1][1], current_node))
                
                # Train the model
                self.optimize_model()
                
                # Check if episode is done
                done = current_node == self.goal_node or len(current_path) > len(self.graph.nodes())
                
                # Add a small delay for visualization sync
                time.sleep(0.05)
            
            # Update best path if this episode had better reward
            if current_node == self.goal_node and episode_reward > self.best_reward:
                self.best_path = current_path[1:]  # Remove the initial None node
                self.best_reward = episode_reward
                print(f"\nNew best path found! Reward: {self.best_reward:.2f}")
                print(f"Path: {' -> '.join(str(node[1]) for node in self.best_path)}")
            
            # Update visualization
            self.training_queue.put({
                'current_edges': current_path[1:],  # Current exploration path
                'past_edges': [],
                'best_path': self.best_path if episode == num_episodes - 1 else None,  # Only show best path at the end
                'training_finished': episode == num_episodes - 1,
                'episode': episode + 1,
                'total_episodes': num_episodes,
                'total_reward': episode_reward,
                'best_reward': self.best_reward
            })
            
            episode_rewards.append(episode_reward)
            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            print(f"\nEpisode {episode + 1} completed - Total Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_reward:.2f}")
            
            # Update exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                     self.exploration_rate * self.exploration_decay)
            
            # Update target network
            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.target_update_counter = 0
                print("Target network updated")
            
            # Add delay between episodes for visualization
            time.sleep(0.2)
        
        # Ensure visualization completes by sending final updates
        for _ in range(3):  # Send multiple final updates to ensure visualization catches up
            time.sleep(0.3)  # Longer delay for final updates
            self.training_queue.put({
                'current_edges': [],
                'past_edges': [],
                'best_path': self.best_path,
                'best_reward': self.best_reward,
                'training_finished': True,
                'episode': num_episodes,
                'total_episodes': num_episodes,
                'total_reward': self.best_reward
            })

def create_graph_figure(graph, positions, highlight_edges=None, current_node=None, explored_edges=None, past_explored_edges=None):
    """Create a Plotly figure for the graph visualization."""
    edge_trace_list = []
    
    # Add all edges in gray first
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    edge_trace_list.append(edge_trace)
    
    # Add explored edges in gold if not training finished
    if explored_edges and not highlight_edges:
        explored_x = []
        explored_y = []
        for edge in explored_edges:
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            explored_x.extend([x0, x1, None])
            explored_y.extend([y0, y1, None])
        
        explored_trace = go.Scatter(
            x=explored_x, y=explored_y,
            line=dict(width=2, color='gold'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_trace_list.append(explored_trace)
    
    # Add best path in green if training is finished
    if highlight_edges:
        best_x = []
        best_y = []
        for edge in highlight_edges:
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            best_x.extend([x0, x1, None])
            best_y.extend([y0, y1, None])
        
        best_trace = go.Scatter(
            x=best_x, y=best_y,
            line=dict(width=3, color='#00FF00'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_trace_list.append(best_trace)
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if node == current_node:
            node_color.append('#FF0000')  # Current node in red
        else:
            node_color.append('#1f77b4')  # Other nodes in default color
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='middle center',
        marker=dict(
            color=node_color,
            size=30,
            line_width=2
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_trace_list + [node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
    )
    
    return fig

def create_dash_app(dq_learner, fixed_positions):
    """
    Create Dash application for Deep Q-Learning visualization.
    """
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1('Deep Q-Learning Path Finding Visualization',
                style={'textAlign': 'center'}),
        
        dcc.Graph(id='graph-visualization',
                 style={'height': '600px'}),
        
        html.Div([
            html.Button('Start Training', id='start-button', n_clicks=0),
            html.Button('Stop Training', id='stop-button', n_clicks=0,
                       style={'margin-left': '10px'})
        ], style={'textAlign': 'center', 'margin': '10px'}),
        
        html.Div(id='path-display',
                style={'textAlign': 'center', 'margin': '10px', 'whiteSpace': 'pre-line'}),
        
        dcc.Interval(id='interval-component',
                    interval=100,
                    n_intervals=0),
        
        html.Div(id='training-status',
                style={'display': 'none'})
    ])
    
    @app.callback(
        [Output('graph-visualization', 'figure'),
         Output('path-display', 'children'),
         Output('training-status', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('start-button', 'n_clicks'),
         Input('stop-button', 'n_clicks')],
        [State('training-status', 'children')]
    )
    def update_visualization(n_intervals, start_clicks, stop_clicks, training_status):
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'start-button' and start_clicks > 0:
            if training_status != 'training':
                # Start training in a separate thread
                training_thread = threading.Thread(
                    target=dq_learner.train,
                    args=(1000,)  # Number of episodes
                )
                training_thread.daemon = True
                training_thread.start()
                return dash.no_update, dash.no_update, 'training'
                
        elif button_id == 'stop-button' and stop_clicks > 0:
            dq_learner.stop_training = True
            return dash.no_update, dash.no_update, 'stopped'
            
        # Get latest training info from queue
        path_text = "Training in progress..."
        current_edges = []
        past_edges = []
        best_path = None
        training_finished = False
        
        try:
            while not dq_learner.training_queue.empty():
                data = dq_learner.training_queue.get_nowait()
                current_edges = data.get('current_edges', [])
                past_edges = data.get('past_edges', [])
                best_path = data.get('best_path', None)
                training_finished = data.get('training_finished', False)
                
                if training_finished and best_path:
                    path_text = (f"Training completed!\n"
                               f"Episodes: {data['episode']}/{data['total_episodes']}\n"
                               f"Best path found: {' -> '.join(str(node[1]) for node in best_path)}\n"
                               f"Total reward: {data['total_reward']:.2f}")
                    training_status = 'finished'
        except queue.Empty:
            pass
            
        # Create graph figure
        fig = create_graph_figure(
            dq_learner.graph,
            fixed_positions,
            highlight_edges=best_path if training_finished else None,
            explored_edges=current_edges if not training_finished else None,
            past_explored_edges=past_edges if not training_finished else None
        )
        
        return fig, path_text, training_status
        
    return app

def create_example_graph():
    """Create a graph using CreationGraph"""
    creation = CreationGraph()
    graph = creation.create_graph(10, 0.3)
    positions = nx.spring_layout(graph, k=1, iterations=50)
    start_node = min(graph.nodes())
    goal_node = max(graph.nodes())
    return graph, positions, start_node, goal_node

def main():
    """Main function to run the Deep Q-Learning visualization"""
    graph, positions, start_node, goal_node = create_example_graph()
    dq_learner = DeepQLearningPathFinder(graph, start_node, goal_node)
    app = create_dash_app(dq_learner, positions)
    app.run_server(debug=True, use_reloader=False)

if __name__ == "__main__":
    main()
