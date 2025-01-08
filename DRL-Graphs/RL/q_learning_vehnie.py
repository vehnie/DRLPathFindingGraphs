import numpy as np
import networkx as nx
import random
import queue
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import threading
import sys
import os

# Add the parent directory to Python path to import from graphfactory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphfactory.creationgraph import CreationGraph

class QLearningPathFinder:
    def __init__(self, graph, start_node, goal_node, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0,
                 no_repeat_nodes=False, no_repeat_edges=False):
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        self.q_table = {}
        self.training_queue = queue.Queue()
        self.stop_training = False
        self.best_path = None
        self.best_reward = float('-inf')
        self.no_repeat_nodes = no_repeat_nodes
        self.no_repeat_edges = no_repeat_edges
        self.initialize_q_table()

    def initialize_q_table(self):
        """Initialize Q-table with zeros for all state-action pairs."""
        for node in self.graph.nodes():
            self.q_table[node] = {
                neighbor: 0.0 
                for neighbor in self.graph.neighbors(node)
            }

    def get_action(self, state, current_path=None):
        """Choose an action using epsilon-greedy policy."""
        if current_path is None:
            current_path = []

        # Get valid neighbors based on rules
        valid_neighbors = []
        for neighbor in self.graph.neighbors(state):
            edge = (state, neighbor)
            # Check node repetition rule
            if self.no_repeat_nodes and any(neighbor == node for _, node in current_path):
                continue
            # Check edge repetition rule
            if self.no_repeat_edges and edge in current_path:
                continue
            valid_neighbors.append(neighbor)

        if not valid_neighbors:  # If no valid moves available
            return None

        if random.random() < self.exploration_rate:
            # Explore: choose a random valid neighbor
            return random.choice(valid_neighbors)
        else:
            # Exploit: choose the best valid action from Q-table
            if state not in self.q_table:
                return None
            # Filter Q-values for valid neighbors only
            valid_q_values = {n: self.q_table[state][n] for n in valid_neighbors if n in self.q_table[state]}
            if not valid_q_values:
                return None
            return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def get_reward(self, state, next_state):
        """Calculate reward for taking action from state to next_state."""
        if next_state == self.goal_node:
            return 100  # High reward for reaching the goal
        elif next_state is None:
            return -100  # Penalty for invalid moves
        else:
            return -1  # Small penalty for each step to encourage shorter paths

    def get_best_path(self):
        """Get the best path using the learned Q-table."""
        if not self.q_table:
            return None
            
        path = []
        current_state = self.start_node
        total_reward = 0
        
        while current_state != self.goal_node:
            if not self.q_table[current_state]:
                return None
                
            # Get the best action from current state
            next_state = max(self.q_table[current_state].items(), key=lambda x: x[1])[0]
            
            # Add edge to path
            edge = (current_state, next_state)
            path.append(edge)
            
            # Update reward
            total_reward += self.get_reward(current_state, next_state)
            
            # Move to next state
            current_state = next_state
            
            # Prevent infinite loops
            if len(path) > len(self.graph.nodes()):
                return None
                
        return path, total_reward

    def train(self, num_episodes=100):
        """Train the agent using Q-learning."""
        current_episode_edges = set()  # Track edges in current episode
        past_episode_edges = set()     # Track edges from past episodes
        
        for episode in range(num_episodes):
            if self.stop_training:
                # Clear the queue before stopping
                while not self.training_queue.empty():
                    try:
                        self.training_queue.get_nowait()
                    except queue.Empty:
                        break
                break
                
            current_state = self.start_node
            total_reward = 0
            current_path = []
            current_episode_edges.clear()  # Clear current episode edges
            
            # Keep track of path and reward for this episode
            episode_complete = False
            episode_path = []
            episode_reward = 0
            
            while current_state != self.goal_node:
                if self.stop_training:  # Check for stop signal within episode
                    break
                    
                # Choose and take action
                next_state = self.get_action(current_state, current_path)
                if next_state is None:  # No valid moves available
                    break  # End this episode and start a new one
                    
                # Record the edge
                edge = (current_state, next_state)
                current_path.append(edge)
                current_episode_edges.add(edge)
                episode_path.append(edge)
                
                # Get reward and update Q-table
                reward = self.get_reward(current_state, next_state)
                episode_reward += reward
                
                # Update Q-value
                if next_state in self.q_table:
                    # Filter valid next actions based on rules
                    valid_next_actions = {}
                    for next_action in self.q_table[next_state]:
                        next_edge = (next_state, next_action)
                        if (not self.no_repeat_nodes or 
                            not any(next_action == node for _, node in current_path)):
                            if (not self.no_repeat_edges or 
                                next_edge not in current_path):
                                valid_next_actions[next_action] = self.q_table[next_state][next_action]
                    
                    best_next_value = max(valid_next_actions.values()) if valid_next_actions else 0
                else:
                    best_next_value = 0
                    
                old_value = self.q_table[current_state][next_state]
                new_value = (1 - self.learning_rate) * old_value + \
                           self.learning_rate * (reward + self.discount_factor * best_next_value)
                self.q_table[current_state][next_state] = new_value
                
                # Move to next state
                current_state = next_state
                
                # Check if we've reached the goal
                if current_state == self.goal_node:
                    episode_complete = True
                    # Update best path if this episode has better reward
                    if episode_reward > self.best_reward:
                        self.best_path = list(episode_path)
                        self.best_reward = episode_reward
                
                # Add current edge to training queue for visualization
                training_finished = episode == num_episodes - 1 and current_state == self.goal_node
                try:
                    self.training_queue.put_nowait({
                        'current_edges': list(current_episode_edges),
                        'past_edges': list(past_episode_edges),
                        'episode': episode + 1,
                        'total_reward': episode_reward,
                        'best_path': self.best_path,
                        'best_reward': self.best_reward,
                        'episode_complete': episode_complete,
                        'training_finished': training_finished
                    })
                except queue.Full:
                    pass  # Skip update if queue is full
                
                time.sleep(0.1)  # Small delay for visualization
            
            if self.stop_training:  # Check for stop signal after episode
                break
                
            # After episode ends, add current edges to past edges
            past_episode_edges.update(current_episode_edges)
            
            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                     self.exploration_rate * self.exploration_decay)
        
        # Send final update with training finished flag
        try:
            self.training_queue.put_nowait({
                'current_edges': list(current_episode_edges),
                'past_edges': list(past_episode_edges),
                'episode': num_episodes,
                'total_reward': episode_reward,
                'best_path': self.best_path,
                'best_reward': self.best_reward,
                'episode_complete': episode_complete,
                'training_finished': True  # Force final update
            })
        except queue.Full:
            pass

    def get_path(self):
        """Get the best path from start to goal using trained Q-table."""
        if not self.q_table:
            return None
            
        path = []
        current_state = self.start_node
        
        while current_state != self.goal_node:
            if current_state not in self.q_table:
                return None
                
            next_state = max(self.q_table[current_state].items(), key=lambda x: x[1])[0]
            path.append((current_state, next_state))
            current_state = next_state
            
            if len(path) > len(self.graph.nodes()):
                return None
                
        return path

# Create Dash App for Q-Learning Visualization
def create_dash_app(q_learner, fixed_positions):
    """
    Create Dash application for Q-Learning visualization.
    
    :param q_learner: QLearningPathFinder instance
    :param fixed_positions: Fixed node positions for consistent layout
    :return: Dash app instance
    """
    app = dash.Dash(__name__)
    
    # Initialize with a figure
    initial_figure = create_graph_figure(
        q_learner.graph,
        fixed_positions,
        current_node=q_learner.start_node
    )
    
    # App layout
    app.layout = html.Div([
        html.H1("Q-Learning Path Finding",
               style={'textAlign': 'center', 'color': 'black', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(
                id='graph-visualization',
                figure=initial_figure,
                style={
                    'height': '800px',
                    'backgroundColor': 'white',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'padding': '10px'
                }
            )
        ], style={
            'margin': '0 auto',
            'maxWidth': '1200px',
            'backgroundColor': 'white'
        }),
        html.Div(id='training-info',
                style={
                    'textAlign': 'center',
                    'marginTop': '20px',
                    'padding': '10px',
                    'backgroundColor': 'white',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px'
                }),
        html.Button('Stop Training', id='stop-training-button', n_clicks=0),
        dcc.Interval(
            id='interval-component',
            interval=100,
            n_intervals=0
        )
    ], style={
        'padding': '20px',
        'minHeight': '100vh',
        'backgroundColor': '#f9f9f9'
    })
    
    # Callback to update the graph and display training info
    @app.callback(
        [Output('graph-visualization', 'figure'),
         Output('training-info', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('stop-training-button', 'n_clicks')]
    )
    def update_graph(n_intervals, n_clicks):
        try:
            # Get the latest training information
            training_info = q_learner.training_queue.get_nowait()
            
            if training_info is None:
                # Training complete, show the best path
                best_path = q_learner.get_best_path()
                best_path_edges = list(zip(best_path[0], best_path[0][1:]))
                fig = create_graph_figure(q_learner.graph, fixed_positions, 
                                       highlight_edges=best_path_edges)
                return fig, "Training Complete! Best path found."
            
            # Show current path from training
            if 'episode' in training_info:
                fig = create_graph_figure(q_learner.graph, fixed_positions, 
                                       current_node=q_learner.start_node)
                info_text = html.Div([
                    html.P([
                        html.Strong("Episode: "), f"{training_info['episode']} | ",
                        html.Strong("Exploration: "), f"{q_learner.exploration_rate:.2f}"
                    ])
                ])
            else:
                current_path = training_info['path']
                current_path_edges = list(zip(current_path, current_path[1:]))
                
                # Create figure with current path and explored edges
                fig = create_graph_figure(
                    q_learner.graph, 
                    fixed_positions, 
                    highlight_edges=current_path_edges,
                    current_node=training_info['current_node'],
                    explored_edges=training_info.get('current_episode_edges', set()),
                    past_explored_edges=training_info.get('past_episode_edges', set())
                )
                
                info_text = html.Div([
                    html.P([
                        html.Strong("Episode: "), f"{training_info['episode']} | ",
                        html.Strong("Path Length: "), f"{len(current_path)} | ",
                        html.Strong("Score: "), f"{training_info['score']} | ",
                        html.Strong("Exploration: "), f"{q_learner.exploration_rate:.2f}"
                    ])
                ])
            
            return fig, info_text
            
        except queue.Empty:
            # No new updates, retain the current figure
            return dash.no_update, dash.no_update
        except Exception as e:
            print(f"Error in update_graph: {e}")
            return dash.no_update, dash.no_update
    
    # Callback to stop training
    @app.callback(
        Output('stop-training-button', 'disabled'),
        [Input('stop-training-button', 'n_clicks')]
    )
    def stop_training(n_clicks):
        if n_clicks > 0:
            q_learner.stop_training = True
            return True
        return False
    
    return app

def create_graph_figure(graph, positions, highlight_edges=None, current_node=None, explored_edges=None, past_explored_edges=None):
    """
    Create a Plotly figure for the graph visualization.

    :param graph: NetworkX graph
    :param positions: Fixed node positions
    :param highlight_edges: List of edges to highlight (current path)
    :param current_node: Current node being explored
    :param explored_edges: List of all explored edges
    :param past_explored_edges: List of all explored edges from past episodes
    :return: Plotly figure
    """
    # Initialize figure with white background
    fig = go.Figure(layout=dict(
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            gridcolor='lightgray',
            range=[-1.5, 1.5]
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showticklabels=True,
            gridcolor='lightgray',
            scaleanchor="x",
            scaleratio=1,
            range=[-1.5, 1.5]
        )
    ))

    # Add edges (gray)
    edge_traces = []
    for edge in graph.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        
        color = '#888'  # default gray
        width = 1
        
        if highlight_edges and edge in highlight_edges:
            color = '#0000ff'  # blue for current path
            width = 3
        elif explored_edges and edge in explored_edges:
            color = '#ffa500'  # orange for explored
            width = 2
        elif past_explored_edges and edge in past_explored_edges:
            color = '#cccccc'  # light gray for past explored
            width = 1
            
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=color),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Add nodes
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        
        # Set node colors and sizes
        if node == current_node:
            node_colors.append('#ff00ff')  # purple for current
            node_sizes.append(30)
        elif node == graph.graph.get('start_node'):
            node_colors.append('#00ff00')  # green for start
            node_sizes.append(25)
        elif node == graph.graph.get('goal_node'):
            node_colors.append('#ff0000')  # red for goal
            node_sizes.append(25)
        else:
            node_colors.append('#1f77b4')  # blue for others
            node_sizes.append(20)

    # Add node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color='white', width=2)
        )
    )

    # Add all traces to figure
    for trace in edge_traces:
        fig.add_trace(trace)
    fig.add_trace(node_trace)

    return fig

def create_example_graph():
    """
    Create a graph using CreationGraph
    """
    # Create a random graph with 3 communities
    graph_factory = CreationGraph(
        predefined=False,
        digraph=True,
        sizes=[4, 3, 3],  # Smaller communities for better visualization
        p_in=0.7,         # Higher intra-community probability
        p_out=0.3         # Higher inter-community probability
    )
    
    # Get the graph and nodes
    result = graph_factory.get_nodes()
    if isinstance(result, tuple):
        G = result[0]
        start_node = result[1]
        goal_node = result[2] if len(result) > 2 else None
    else:
        G = result
        nodes = list(G.nodes())
        start_node, goal_node = random.sample(nodes, 2)
    
    # Store start and goal nodes in graph attributes
    G.graph['start_node'] = start_node
    G.graph['goal_node'] = goal_node
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Start node: {start_node}, Goal node: {goal_node}")
    
    return G

def main():
    print("Starting the Q-Learning visualization...")
    
    # Create the graph
    print("Creating example graph...")
    graph = create_example_graph()
    
    # Get start and goal nodes from graph attributes
    start_node = graph.graph['start_node']
    goal_node = graph.graph['goal_node']
    
    # Generate fixed node positions with more spacing
    print("Generating node positions...")
    fixed_positions = nx.spring_layout(
        graph,
        k=2.0,          # Increase spacing between nodes
        iterations=50,   # More iterations for better layout
        seed=42         # Fixed seed for reproducibility
    )
    
    # Scale positions to fit in [-1, 1] range
    max_pos = max(max(abs(x), abs(y)) for x, y in fixed_positions.values())
    fixed_positions = {node: (x/max_pos, y/max_pos) 
                      for node, (x, y) in fixed_positions.items()}
    
    # Initialize Q-learning agent
    print("Initializing Q-learning agent...")
    q_learner = QLearningPathFinder(
        graph, 
        start_node=start_node,
        goal_node=goal_node,
        no_repeat_nodes=True,
        no_repeat_edges=True
    )
    
    # Create Dash app
    print("Creating Dash app...")
    app = create_dash_app(q_learner, fixed_positions)
    
    # Start training in a separate thread
    print("Starting training thread...")
    training_thread = threading.Thread(target=q_learner.train, kwargs={'num_episodes': 100})
    training_thread.daemon = True
    training_thread.start()
    
    # Run the Dash app
    print("Starting Dash server...")
    app.run_server(debug=False, port=8050)

if __name__ == "__main__":
    main()