import dash
from dash import html, dcc, Input, Output, State, no_update
import dash_cytoscape as cyto
import networkx as nx
import threading
import queue
import random
import time
import json
import base64
from RL.q_learning_vehnie import QLearningPathFinder
from RL.deep_q_learning import DeepQLearningPathFinder
from pathfinding.pathfinder import PathFinder
from graphfactory.creationgraph import GraphFactory
from pathfinding.pathfinder import PathFinder

class App:
    def __init__(self):
        """Initialize the application."""
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.graph = None
        self.node_positions = None
        self.start_node = None
        self.end_node = None
        self.agent = None
        self.training_thread = None
        self.stop_event = threading.Event()
        self.best_path = None
        self.best_reward = float('-inf')
        self.edges = []
        self.nodes = []
        self.previous_start_node = None  # Track previous start node
        self.previous_end_node = None    # Track previous end node
        self.previous_params = None      # Store previous graph parameters
        self.used_node_pairs = set()     # Keep track of previously used start-end node pairs
        self.stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'background-color': '#66b2ff',
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': 30,
                    'height': 30
                }
            },
            {
                'selector': 'node[type="start"]',
                'style': {
                    'background-color': '#4CAF50'  # Green for start node
                }
            },
            {
                'selector': 'node[type="end"]',
                'style': {
                    'background-color': '#f44336'  # Red for end node
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'width': 2,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            },
            {
                'selector': '.current',
                'style': {
                    'line-color': '#ffd700',  # Gold color for current path
                    'target-arrow-color': '#ffd700',
                    'width': 3,
                    'z-index': 1
                }
            },
            {
                'selector': '.explored',
                'style': {
                    'line-color': '#808080',  # Gray for explored paths
                    'target-arrow-color': '#808080',
                    'width': 2
                }
            },
            {
                'selector': '.best',
                'style': {
                    'line-color': '#00ff00',  # Bright green for best path
                    'target-arrow-color': '#00ff00',
                    'width': 4,
                    'z-index': 2  # Ensure best path is drawn on top
                }
            }
        ]
        self.setup_layout()
        self.setup_callbacks()

    def get_node_positions(self):
        """Calculate node positions using a deterministic layout."""
        if not self.graph:
            return {}
            
        # Use spring_layout with fixed seed for consistency
        pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
        # Convert numpy arrays to lists for JSON serialization
        return {node: {'x': float(coords[0]), 'y': float(coords[1])} 
                for node, coords in pos.items()}

    def setup_layout(self):
        """Setup the Dash app layout."""
        self.app.layout = html.Div([
            html.H1("Graph Path Finding", style={'textAlign': 'center'}),
            html.Div([
                # Left side: Graph
                html.Div([
                    cyto.Cytoscape(
                        id='cytoscape',
                        elements=[],
                        style={'width': '100%', 'height': '800px'},
                        layout={'name': 'preset'},  # Use preset layout for fixed positions
                        stylesheet=self.stylesheet
                    )
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Right side: Controls
                html.Div([
                    html.H3("Graph Generation"),
                    dcc.Dropdown(
                        id='graph-type',
                        options=[
                            {'label': 'Erdos-Renyi', 'value': 'erdos_renyi'},
                            {'label': 'Barabasi-Albert', 'value': 'barabasi_albert'},
                            {'label': 'Scale-Free', 'value': 'scale_free'}
                        ],
                        value='erdos_renyi'
                    ),
                    
                    # Graph file upload and download
                    html.Div([
                        dcc.Upload(
                            id='upload-graph',
                            children=html.Button('Upload Graph', 
                                style={
                                    'width': '45%',
                                    'backgroundColor': '#3498db',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px',
                                    'borderRadius': '5px'
                                }
                            ),
                            multiple=False
                        ),
                        html.Button(
                            'Download Graph',
                            id='download-graph',
                            style={
                                'width': '45%',
                                'backgroundColor': '#3498db',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'marginLeft': '10%'
                            }
                        ),
                        dcc.Download(id='download-graph-file')
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '10px', 'marginBottom': '20px'}),
                    
                    # Common parameters
                    html.Div([
                        html.Label("Number of Nodes"),
                        dcc.Input(id='num-nodes', type='number', value=10, min=2)
                    ], style={'marginTop': '10px'}),
                    
                    # Erdos-Renyi parameters
                    html.Div(id='er-params', children=[
                        html.Label("Edge Probability"),
                        dcc.Input(id='er-prob', type='number', value=0.2, min=0.1, max=0.9, step=0.1)
                    ], style={'marginTop': '10px'}),
                    
                    # Barabasi-Albert parameters
                    html.Div(id='ba-params', children=[
                        html.Label("Edges per New Node"),
                        dcc.Input(id='ba-edges', type='number', value=3, min=1, max=10)
                    ], style={'marginTop': '10px', 'display': 'none'}),
                    
                    # Scale-free parameters
                    html.Div(id='sf-params', children=[
                        html.Label("Alpha (Out-degree)"),
                        dcc.Input(id='sf-alpha', type='number', value=2.0, min=0.1, max=5.0, step=0.1),
                        html.Label("Beta (In-degree)"),
                        dcc.Input(id='sf-beta', type='number', value=0.0, min=0.0, max=5.0, step=0.1),
                        html.Label("Gamma (Initial Attractiveness)"),
                        dcc.Input(id='sf-gamma', type='number', value=1.0, min=0.1, max=5.0, step=0.1)
                    ], style={'marginTop': '10px', 'display': 'none'}),
                    
                    html.Button(
                        'Generate Graph',
                        id='generate-graph',
                        style={
                            'width': '100%',
                            'marginTop': '20px',
                            'marginBottom': '20px',
                            'backgroundColor': '#3498db',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px',
                            'borderRadius': '5px'
                        }
                    ),
                    
                    html.H3("Algorithm Controls"),
                    html.Label("Layout:"),
                    dcc.Dropdown(
                        id='layout-dropdown',
                        options=[
                            {'label': 'Cose', 'value': 'cose'},
                            {'label': 'Circle', 'value': 'circle'},
                            {'label': 'Grid', 'value': 'grid'},
                            {'label': 'Random', 'value': 'random'}
                        ],
                        value='cose'
                    ),
                    html.Br(),
                    html.Label("Algorithm:"),
                    dcc.Dropdown(
                        id='algorithm-type',
                        options=[
                            {'label': 'Deep Q-Learning', 'value': 'deep_q_learning'},
                            {'label': 'Q-Learning', 'value': 'q_learning'},
                            {'label': 'A*', 'value': 'astar'},
                            {'label': 'Dijkstra', 'value': 'dijkstra'}
                        ],
                        value='deep_q_learning'
                    ),
                    html.Br(),
                    
                    # Q-Learning parameters
                    html.Div([
                        html.Label("Learning Rate"),
                        dcc.Input(id='learning-rate', type='number', value=0.1, min=0.01, max=1.0, step=0.01),
                        html.Label("Discount Factor"),
                        dcc.Input(id='discount-factor', type='number', value=0.9, min=0.01, max=1.0, step=0.01),
                        html.Label("Initial Exploration Rate"),
                        dcc.Input(id='exploration-rate', type='number', value=1.0, min=0.01, max=1.0, step=0.01),
                        html.Label("Number of Episodes"),
                        dcc.Input(id='num-episodes', type='number', value=100, min=1, step=1),
                        html.Label("Path Rules"),
                        dcc.Checklist(
                            id='path-rules',
                            options=[
                                {'label': ' No repeated nodes', 'value': 'no_repeat_nodes'},
                                {'label': ' No repeated edges', 'value': 'no_repeat_edges'}
                            ],
                            value=[],
                            style={'marginTop': '10px'}
                        ),
                    ], id='q-learning-params', style={'marginTop': '10px'}),
                    
                    # A* parameters
                    html.Div([
                        html.Label("Heuristic Function"),
                        dcc.Dropdown(
                            id='astar-heuristic',
                            options=[
                                {'label': 'Manhattan Distance', 'value': 'manhattan'},
                                {'label': 'Euclidean Distance', 'value': 'euclidean'},
                                {'label': 'Dijkstra (h=0)', 'value': 'dijkstra'}
                            ],
                            value='manhattan'
                        ),
                        html.Label("Weight (w ≥ 1)"),
                        dcc.Input(id='astar-weight', type='number', value=1.0, min=1.0, max=5.0, step=0.1),
                    ], id='astar-params', style={'marginTop': '10px', 'display': 'none'}),
                    
                    # Dijkstra doesn't need parameters as it's deterministic
                    html.Div([
                        html.P("Dijkstra's algorithm is deterministic and doesn't require additional parameters."),
                    ], id='dijkstra-params', style={'marginTop': '10px', 'display': 'none'}),
                    
                    html.Div([
                        html.Button(
                            'Start Training',
                            id='start-button',
                            style={
                                'width': '45%',
                                'marginRight': '10%',
                                'backgroundColor': '#2ecc71',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px',
                                'borderRadius': '5px'
                            }
                        ),
                        html.Button(
                            'Stop Training',
                            id='stop-button',
                            style={
                                'width': '45%',
                                'backgroundColor': '#e74c3c',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px',
                                'borderRadius': '5px'
                            }
                        ),
                    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                    html.Br(),
                    html.Div(
                        id='status-text',
                        style={
                            'marginTop': '10px',
                            'padding': '10px',
                            'backgroundColor': '#f8f9fa',
                            'borderRadius': '5px',
                            'whiteSpace': 'pre-wrap',
                            'fontFamily': 'monospace'
                        }
                    ),
                    dcc.Interval(id='interval-component', interval=100, n_intervals=0)
                ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'})
            ]),
        ])

    def setup_callbacks(self):
        """Setup the Dash app callbacks."""
        @self.app.callback(
            [Output('er-params', 'style'),
             Output('ba-params', 'style'),
             Output('sf-params', 'style')],
            [Input('graph-type', 'value')]
        )
        def toggle_params(graph_type):
            er_style = {'marginTop': '10px', 'display': 'block' if graph_type == 'erdos_renyi' else 'none'}
            ba_style = {'marginTop': '10px', 'display': 'block' if graph_type == 'barabasi_albert' else 'none'}
            sf_style = {'marginTop': '10px', 'display': 'block' if graph_type == 'scale_free' else 'none'}
            return er_style, ba_style, sf_style
        
        @self.app.callback(
            [Output('q-learning-params', 'style'),
             Output('astar-params', 'style'),
             Output('dijkstra-params', 'style')],
            [Input('algorithm-type', 'value')]
        )
        def toggle_algorithm_params(algorithm_type):
            q_learning_style = {'marginTop': '10px', 'display': 'block' if algorithm_type in ['q_learning', 'deep_q_learning'] else 'none'}
            astar_style = {'marginTop': '10px', 'display': 'block' if algorithm_type == 'astar' else 'none'}
            dijkstra_style = {'marginTop': '10px', 'display': 'block' if algorithm_type == 'dijkstra' else 'none'}
            return q_learning_style, astar_style, dijkstra_style
        
        @self.app.callback(
            [Output('cytoscape', 'elements'),
             Output('status-text', 'children'),
             Output('start-button', 'disabled'),
             Output('stop-button', 'disabled')],
            [Input('generate-graph', 'n_clicks'),
             Input('interval-component', 'n_intervals'),
             Input('start-button', 'n_clicks'),
             Input('stop-button', 'n_clicks'),
             Input('algorithm-type', 'value'),
             Input('learning-rate', 'value'),
             Input('discount-factor', 'value'),
             Input('exploration-rate', 'value'),
             Input('astar-heuristic', 'value'),
             Input('astar-weight', 'value'),
             Input('num-episodes', 'value'),
             Input('path-rules', 'value')],
            [State('graph-type', 'value'),
             State('num-nodes', 'value'),
             State('er-prob', 'value'),
             State('ba-edges', 'value'),
             State('sf-alpha', 'value'),
             State('sf-beta', 'value'),
             State('sf-gamma', 'value')]
        )
        def update_graph_and_training(generate_clicks, n_intervals, start_clicks, stop_clicks,
                                    algorithm_type, learning_rate, discount_factor, exploration_rate,
                                    astar_heuristic, astar_weight, num_episodes, path_rules,
                                    graph_type, num_nodes, er_prob, ba_edges,
                                    sf_alpha, sf_beta, sf_gamma):
            ctx = dash.callback_context
            if not ctx.triggered:
                return no_update, no_update, True, True

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if trigger_id == 'generate-graph' and generate_clicks:
                # Validate inputs
                try:
                    num_nodes = int(num_nodes)  # Convert to int explicitly
                    er_prob = float(er_prob)    # Convert to float explicitly
                except (TypeError, ValueError):
                    return [], "Error: Invalid input values", True, True
                    
                if num_nodes is None or num_nodes < 2:
                    return [], "Error: Number of nodes must be at least 2", True, True
                if er_prob is None or not (0 <= er_prob <= 1):
                    return [], "Error: Edge probability must be between 0 and 1", True, True
                
                # Reset everything before creating new graph
                self.graph = None
                self.edges = []
                self.nodes = []
                self.agent = None
                self.used_node_pairs = set()  # Clear used pairs for new graph
                if self.training_thread and self.training_thread.is_alive():
                    self.stop_event.set()
                    self.training_thread.join()
                self.training_thread = None
                self.start_node = None
                self.end_node = None
                self.previous_start_node = None
                self.previous_end_node = None
                self.best_path = None
                self.best_reward = float('-inf')
                
                try:
                    # Generate new graph
                    factory = GraphFactory()
                    if graph_type == 'erdos_renyi':
                        self.graph = factory.create_erdos_renyi(num_nodes, er_prob)
                    elif graph_type == 'barabasi_albert':
                        self.graph = factory.create_barabasi_albert(num_nodes, ba_edges)
                    elif graph_type == 'scale_free':
                        self.graph = factory.create_scale_free(num_nodes, alpha=sf_alpha, beta=sf_beta, gamma=sf_gamma)
                        
                    # Get node positions from graph attributes
                    self.node_positions = nx.get_node_attributes(self.graph, 'pos')
                    if not self.node_positions:  # Fallback if positions not set
                        self.node_positions = self.get_node_positions()
                except ValueError as e:
                    return [], f"Error: {str(e)}", True, True

                # Instead of taking largest strongly connected component,
                # add edges to ensure strong connectivity while preserving all nodes
                if len(self.graph) > 0:
                    # Get strongly connected components
                    components = list(nx.strongly_connected_components(self.graph))
                    if len(components) > 1:
                        # Connect components with directed edges
                        nodes_to_connect = [min(comp) for comp in components]
                        for i in range(len(nodes_to_connect)-1):
                            self.graph.add_edge(nodes_to_connect[i], nodes_to_connect[i+1])

                # Select random start and end nodes
                nodes = list(self.graph.nodes())
                if len(nodes) < 2:
                    return [], "Error: Graph must have at least 2 nodes", True, True

                # For 2-node graphs, just alternate between the two possible pairs
                if len(nodes) == 2:
                    if (0, 1) not in self.used_node_pairs:
                        self.start_node = 0
                        self.end_node = 1
                        self.used_node_pairs.add((0, 1))
                    else:
                        self.start_node = 1
                        self.end_node = 0
                        self.used_node_pairs = {(1, 0)}  # Reset to only this pair
                else:
                    # For larger graphs, try to find unused pairs
                    max_attempts = 100
                    found_valid_pair = False
                    
                    for _ in range(max_attempts):
                        candidate_start = random.choice(nodes)
                        remaining_nodes = [n for n in nodes if n != candidate_start]
                        candidate_end = random.choice(remaining_nodes)
                        
                        pair = (candidate_start, candidate_end)
                        if pair not in self.used_node_pairs and nx.has_path(self.graph, candidate_start, candidate_end):
                            self.start_node = candidate_start
                            self.end_node = candidate_end
                            self.used_node_pairs.add(pair)
                            found_valid_pair = True
                            break
                    
                    if not found_valid_pair:
                        # If we can't find an unused pair, clear history and try one more time
                        self.used_node_pairs.clear()
                        candidate_start = random.choice(nodes)
                        remaining_nodes = [n for n in nodes if n != candidate_start]
                        candidate_end = random.choice(remaining_nodes)
                        if nx.has_path(self.graph, candidate_start, candidate_end):
                            self.start_node = candidate_start
                            self.end_node = candidate_end
                            self.used_node_pairs.add((candidate_start, candidate_end))
                            found_valid_pair = True
                
                if not found_valid_pair:
                    return [], "Error: Could not find valid start and end nodes. Try regenerating the graph.", True, True

                # Update elements for visualization
                elements = []
                if self.graph:
                    # Calculate node positions if not already set
                    if not self.node_positions:
                        self.node_positions = self.get_node_positions()
                    
                    # Add nodes with their fixed positions
                    for node in self.graph.nodes():
                        node_type = 'default'
                        if node == self.start_node:
                            node_type = 'start'
                        elif node == self.end_node:
                            node_type = 'end'
                        
                        elements.append({
                            'data': {
                                'id': str(node),
                                'label': str(node),
                                'type': node_type
                            }
                        })
                    
                    # Add edges with default styling for new graphs
                    for edge in self.graph.edges():
                        elements.append({
                            'data': {
                                'id': f"edge_{edge[0]}_{edge[1]}",  # Unique edge ID
                                'source': str(edge[0]),
                                'target': str(edge[1]),
                                'type': 'default'  # Explicitly set type to default
                            },
                            'classes': ''  # Clear any previous classes
                        })
                    info_text = ""
                return elements, f"Graph generated with start node {self.start_node} (green) and goal node {self.end_node} (red)", False, True

            elif trigger_id == 'start-button' and start_clicks:
                if not self.graph or self.start_node is None or self.end_node is None:
                    return no_update, "Please generate a graph first", True, True

                # Stop any existing training
                if self.training_thread and self.training_thread.is_alive():
                    # Set stop flag for Q-learning agent
                    if self.agent:
                        self.agent.stop_training = True
                    # Set stop event for thread
                    self.stop_event.set()
                    # Wait for thread to finish
                    self.training_thread.join(timeout=1.0)
                    # Clear any remaining items in the queue
                    while not (self.agent.training_queue if hasattr(self.agent, 'training_queue') else queue.Queue()).empty():
                        try:
                            self.agent.training_queue.get_nowait()
                        except queue.Empty:
                            break

                if algorithm_type == 'deep_q_learning':
                    # Initialize new Deep Q-Learning agent with current parameters
                    self.agent = DeepQLearningPathFinder(
                        self.graph,
                        self.start_node,
                        self.end_node,
                        learning_rate=float(learning_rate),
                        discount_factor=float(discount_factor),
                        exploration_rate=float(exploration_rate),
                        no_repeat_nodes='no_repeat_nodes' in path_rules,
                        no_repeat_edges='no_repeat_edges' in path_rules
                    )
                    
                    # Reset stop flags
                    self.stop_event.clear()
                    self.agent.stop_training = False
                    
                    # Start training in a separate thread
                    self.training_thread = threading.Thread(
                        target=self.agent.train,
                        kwargs={'num_episodes': int(num_episodes)}
                    )
                elif algorithm_type == 'q_learning':
                    # Initialize new Q-Learning agent with current parameters
                    self.agent = QLearningPathFinder(
                        self.graph,
                        self.start_node,
                        self.end_node,
                        learning_rate=float(learning_rate),
                        discount_factor=float(discount_factor),
                        exploration_rate=float(exploration_rate),
                        no_repeat_nodes='no_repeat_nodes' in path_rules,
                        no_repeat_edges='no_repeat_edges' in path_rules
                    )
                    
                    # Reset stop flags
                    self.stop_event.clear()
                    self.agent.stop_training = False
                    
                    # Start training in a separate thread
                    self.training_thread = threading.Thread(
                        target=self.agent.train,
                        kwargs={'num_episodes': int(num_episodes)}
                    )
                elif algorithm_type == 'astar':
                    # Initialize A* pathfinder
                    self.agent = PathFinder(
                        self.graph,
                        self.start_node,
                        self.end_node
                    )
                    
                    # Reset stop flags
                    self.stop_event.clear()
                    self.agent.stop_training = False
                    
                    # Start A* in a separate thread
                    self.training_thread = threading.Thread(
                        target=self.agent.run_astar
                    )
                elif algorithm_type == 'dijkstra':
                    # Initialize Dijkstra pathfinder
                    self.agent = PathFinder(
                        self.graph,
                        self.start_node,
                        self.end_node
                    )
                    
                    # Reset stop flags
                    self.stop_event.clear()
                    self.agent.stop_training = False
                    
                    # Start Dijkstra in a separate thread
                    self.training_thread = threading.Thread(
                        target=self.agent.run_dijkstra
                    )
                
                self.training_thread.start()
                return no_update, "Training started with new parameters...", True, False

            elif trigger_id == 'stop-button' and stop_clicks:
                if self.training_thread and self.training_thread.is_alive():
                    # Set stop flag for Q-learning agent
                    if self.agent:
                        self.agent.stop_training = True
                    # Set stop event for thread
                    self.stop_event.set()
                    # Wait for thread to finish
                    self.training_thread.join(timeout=1.0)
                    # Clear the training queue
                    while not self.agent.training_queue.empty():
                        try:
                            self.agent.training_queue.get_nowait()
                        except queue.Empty:
                            break
                    return no_update, "Training stopped. You can adjust parameters and start training again.", False, True
                return no_update, no_update, False, True

            elif trigger_id == 'interval-component':
                if not self.agent or not self.training_thread or not self.training_thread.is_alive():
                    return no_update, no_update, False, True

                # Get updates from the training queue
                try:
                    update = self.agent.training_queue.get_nowait()
                except queue.Empty:
                    return no_update, no_update, True, False

                # Update visualization based on algorithm type
                elements = []
                
                # Add nodes
                for node in self.graph.nodes():
                    node_type = 'default'
                    if node == self.start_node:
                        node_type = 'start'
                    elif node == self.end_node:
                        node_type = 'end'
                    elements.append({
                        'data': {
                            'id': str(node),
                            'label': str(node),
                            'type': node_type
                        }
                    })

                # Add edges with appropriate styling
                for edge in self.graph.edges():
                    edge_classes = []
                    training_finished = update.get('training_finished', False)
                    
                    if isinstance(self.agent, QLearningPathFinder) or isinstance(self.agent, DeepQLearningPathFinder):
                        # Q-Learning specific visualization
                        if training_finished and update.get('best_path') and (edge[0], edge[1]) in update['best_path']:
                            # Show best path in green when training is finished
                            edge_classes.append('best')
                        elif edge in update.get('current_edges', []):
                            # Show current exploration in gold
                            edge_classes.append('current')
                        elif edge in update.get('past_edges', []):
                            # Show past explorations in gray
                            edge_classes.append('explored')
                    else:
                        # A* and Dijkstra visualization
                        if training_finished and update.get('final_path') and edge in update['final_path']:
                            # Show final path in green
                            edge_classes.append('best')
                        elif edge in update.get('current_episode_edges', []):
                            # Show current path in gold
                            edge_classes.append('current')
                        elif edge in update.get('past_episode_edges', []):
                            # Show explored edges in gray
                            edge_classes.append('explored')

                    elements.append({
                        'data': {
                            'id': f"edge_{edge[0]}_{edge[1]}",
                            'source': str(edge[0]),
                            'target': str(edge[1])
                        },
                        'classes': ' '.join(edge_classes)
                    })

                # Update status text based on algorithm type
                if isinstance(self.agent, QLearningPathFinder) or isinstance(self.agent, DeepQLearningPathFinder):
                    # Update best path and reward if necessary
                    if update.get('best_path') and update.get('best_reward', float('-inf')) > self.best_reward:
                        self.best_path = update['best_path']
                        self.best_reward = update['best_reward']

                    if update.get('training_finished'):
                        # Format status text for completed training
                        path_nodes = []
                        current_node = self.start_node
                        path_nodes.append(str(current_node))
                        for edge in self.best_path:
                            path_nodes.append(str(edge[1]))
                        status = (
                            f"Training finished!\n"
                            f"Best path found: {' -> '.join(path_nodes)}\n"
                            f"Best reward: {self.best_reward:.2f}\n"
                            f"Current Reward: {float(update.get('total_reward', 0.0)):.2f}\n"
                            f"Episodes completed: {update.get('episode', 0)}/{update.get('total_episodes', 0)}"
                        )
                    else:
                        # Format status text for ongoing training
                        status = (
                            f"Episode: {update.get('episode', 'N/A')}\n"
                            f"Current Reward: {float(update.get('total_reward', 0.0)):.2f}\n"
                            f"Best Reward: {self.best_reward:.2f}"
                        )
                else:
                    current_status = f"Exploring... Current score: {update.get('score', 'N/A')}"
                    if training_finished and update.get('final_path'):
                        # Show the final path for A* and Dijkstra
                        path_nodes = []
                        current_node = self.start_node
                        path_nodes.append(str(current_node))
                        for edge in update['final_path']:
                            path_nodes.append(str(edge[1]))
                        status = (
                            f"{current_status}\n\n"
                            f"Path found!\n"
                            f"Path length: {len(update['final_path'])}\n"
                            f"Total cost: {-update.get('score', 'N/A')}\n"
                            f"Path: {' -> '.join(path_nodes)}"
                        )
                    elif training_finished and update.get('final_path') is None:
                        status = f"{current_status}\n\nNo path found to goal node!"
                    else:
                        status = current_status

                return elements, status, True, False

            return no_update, no_update, no_update, no_update

        @self.app.callback(
            [Output('cytoscape', 'elements', allow_duplicate=True),
             Output('status-text', 'children', allow_duplicate=True)],
            Input('upload-graph', 'contents'),
            State('upload-graph', 'filename'),
            prevent_initial_call=True
        )
        def upload_graph(contents, filename):
            if contents is None:
                return no_update, no_update
            
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            try:
                # Load graph data from JSON
                graph_data = json.loads(decoded.decode('utf-8'))
                
                # Create new NetworkX graph
                self.graph = nx.DiGraph()
                self.graph.add_nodes_from(graph_data['nodes'])
                self.graph.add_edges_from(graph_data['edges'])
                self.start_node = graph_data['start_node']
                self.end_node = graph_data['goal_node']
                
                # Calculate node positions
                self.node_positions = self.get_node_positions()
                
                # Create elements for cytoscape
                elements = []
                
                # Add nodes
                for node in self.graph.nodes():
                    node_type = 'default'
                    if node == self.start_node:
                        node_type = 'start'
                    elif node == self.end_node:
                        node_type = 'end'
                    elements.append({
                        'data': {
                            'id': str(node),
                            'label': str(node),
                            'type': node_type
                        }
                    })
                
                # Add edges
                for edge in self.graph.edges():
                    elements.append({
                        'data': {
                            'id': f"edge_{edge[0]}_{edge[1]}",
                            'source': str(edge[0]),
                            'target': str(edge[1])
                        },
                        'classes': ''
                    })
                
                return elements, f"Graph loaded from {filename}"
            except Exception as e:
                return no_update, f"Error loading graph: {str(e)}"

        @self.app.callback(
            [Output('cytoscape', 'elements', allow_duplicate=True),
             Output('status-text', 'children', allow_duplicate=True)],
            [Input('algorithm-type', 'value')],
            [State('cytoscape', 'elements')],
            prevent_initial_call=True
        )
        def reset_visualization(algorithm_type, current_elements):
            """Reset the visualization when algorithm type changes."""
            if not current_elements:
                return no_update, no_update
                
            # Reset all edge colors by removing classes
            elements = []
            for element in current_elements:
                if element['data'].get('source'):  # It's an edge
                    elements.append({
                        'data': {
                            'id': element['data']['id'],
                            'source': element['data']['source'],
                            'target': element['data']['target']
                        },
                        'classes': ''  # Clear all classes
                    })
                else:  # It's a node
                    elements.append(element)  # Keep node styling
                    
            return elements, f"Selected algorithm: {algorithm_type}"

        @self.app.callback(
            Output('cytoscape', 'layout'),
            [Input('layout-dropdown', 'value')]
        )
        def update_layout(layout):
            return {'name': layout}

        @self.app.callback(
            Output('download-graph-file', 'data'),
            Input('download-graph', 'n_clicks'),
            prevent_initial_call=True
        )
        def download_graph(n_clicks):
            if not self.graph:
                return None
            
            # Create graph data dictionary
            graph_data = {
                'nodes': list(self.graph.nodes()),
                'edges': list(self.graph.edges()),
                'start_node': self.start_node,
                'goal_node': self.end_node
            }
            
            return dict(
                content=json.dumps(graph_data, indent=2),
                filename='graph.json'
            )

    def run(self):
        """Run the Dash app."""
        self.app.run_server(debug=True)
        
if __name__ == '__main__':
    app = App()
    app.run()
