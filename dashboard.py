import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
from plot_utils import *

# Load data
data_objects = pd.read_csv('data_objects.csv')
data_trajectories = pd.read_csv('data_trajectories.csv')

# Prepare Charts
data_bump, color_map = prepare_bump(data_objects)
metadata_objects, features_objects = prepare_state_space(data_objects, metadata_to=7)

# Prepare DAG
global dag_data
dag_data = prepare_DAG('data_trajectories.csv', n_trajectories=100)

app = dash.Dash(__name__)

# Load extra layouts for cytoscape
cyto.load_extra_layouts()

metrics = ["reward_ranked", "frequency", "reward1", "reward2",
           "reward3", "reward_total", "Custom Reward"]

flow_options = ["flow_forward", "flow_backward",
                "flow_forward_change", "flow_backward_change"]

# Layout
app.layout = html.Div([
    dcc.Store(id="prev-node-truncation", data=0),
    html.Div([
        html.Div([
            # ---------- TOP LEFT ----------
            html.Div([
                html.Label("Metric"),
                dcc.Dropdown(
                    id="metric",
                    options=[{"label": m, "value": m} for m in metrics],
                    value="reward_ranked",
                    clearable=False
                ),
                html.Div(
                    id="custom-weights",
                    children=[
                        html.Div([
                            html.Div("Method", style={"textAlign": "center", "flex": 1}),
                            html.Div("Weight1", style={"textAlign": "center", "flex": 1}),
                            html.Div("Weight2", style={"textAlign": "center", "flex": 1}),
                            html.Div("Weight3", style={"textAlign": "center", "flex": 1}),
                        ], style={"display": "flex", "gap": "20px"}),

                        html.Div([
                            html.Div(dcc.Dropdown(
                                id="method",
                                options=["addition", "multiplication"],
                                value="addition",
                                clearable=False
                            ), style={"flex": 1}),
                            html.Div(dcc.Slider(id="w1", min=0, max=2, value=1),
                                     style={"flex": 1}),
                            html.Div(dcc.Slider(id="w2", min=0, max=2, value=1),
                                     style={"flex": 1}),
                            html.Div(dcc.Slider(id="w3", min=0, max=2, value=1),
                                     style={"flex": 1}),
                        ], style={"display": "flex", "gap": "20px", "marginBottom": "15px"}),
                    ],
                    style={"display": "none", "marginTop": "15px"}
                ),
                html.Div(
                    dcc.Graph(id="bumpchart"), style={"height": "100%", "width": "100%"}),
            ], style={
                "flex": 1, "border": "1px solid #ddd", "padding": "5px",
                "height": "49vh", "box-sizing": "border-box", "overflow": "hidden",
            }),

            # ---------- TOP RIGHT ----------
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Projection method", style={"textAlign": "center", "flex": 1}),
                        html.Div("n_neighbors (UMAP) / perplexity (t-SNE)",
                                 style={"textAlign": "center", "flex": 1}),
                    ], style={"display": "flex", "gap": "20px"}),
                    html.Div([
                        html.Div(
                            dcc.Dropdown(
                                id="projection-method",
                                options=[
                                    {"label": "UMAP", "value": "umap"},
                                    {"label": "t-SNE", "value": "tsne"}
                                ],
                                value="umap",
                                clearable=False
                            ),
                            style={"flex": 1}
                        ),
                        html.Div(
                            dcc.Slider(
                                id="projection-param",
                                min=5,
                                max=50,
                                step=1,
                                value=15,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            style={"flex": 1}
                        )
                    ], style={"display": "flex", "gap": "20px"})
                ], style={"display": "block", "marginTop": "15px"}),
                dcc.Graph(id="state-space-plot")
            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "box-sizing": "border-box"
            }),
        ], style={
            "display": "flex", "flex-direction": "row", "width": "100%",
        }),

        html.Div([
            # ---------- BOTTOM LEFT (DAG) ----------
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Flow attribute", style={"textAlign": "center", "flex": 1}),
                        html.Div("Trajectories", style={"textAlign": "center", "flex": 1}),
                        html.Div("Edges (%)", style={"textAlign": "center", "flex": 1}),
                        html.Div("Layout", style={"textAlign": "center", "flex": 1}),
                    ], style={"display": "flex", "gap": "20px"}),
                    html.Div([
                        html.Div(
                            dcc.Dropdown(
                                id="flow-attr",
                                options=[{"label": f, "value": f} for f in flow_options],
                                value="flow_forward",
                                clearable=False
                            ),
                            style={"flex": 1}
                        ),
                        html.Div(
                            dcc.Slider(
                                id="trajectory-truncation",
                                min=1,
                                max=200,
                                step=5,
                                value=10,
                                marks={0: '0', 100: '100', 200: '200'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            style={"flex": 1}
                        ),
                        html.Div(
                            dcc.Slider(
                                id="edge-truncation",
                                min=0,
                                max=100,
                                step=5,
                                value=0,
                                marks={0: '0%', 50: '50%', 100: '100%'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            style={"flex": 1}
                        ),
                        html.Div(
                            dcc.Dropdown(
                                id="dag-layout",
                                options=[
                                    {"label": "Klay", "value": "klay"},
                                    {"label": "Dagre", "value": "dagre"},
                                    {"label": "Breadthfirst", "value": "breadthfirst"}
                                ],
                                value="klay",
                                clearable=False
                            ),
                            style={"flex": 1}
                        )
                    ], style={"display": "flex", "gap": "20px", "marginBottom": "10px"})
                ], style={"display": "block", "marginTop": "10px"}),

                html.Div([
                    cyto.Cytoscape(
                        id='dag-graph',
                        layout={
                            'name': 'klay',
                            'directed': True,
                            'spacingFactor': 1.0,
                            'animate': False
                        },
                        style={'flex': '1',  'height': '42vh', 'width': '0px'},
                        elements=[],
                        stylesheet=[]
                    ),
                    dcc.Graph(
                        id='dag-legend',
                        style={'width': '75px', 'height': '42vh', 'flex': '0 0 75px'}
                    )
                ], style={"display": "flex", "flex-direction": "row", "width": "100%"})
            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "box-sizing": "border-box",
                "overflow": "hidden"
            }),

            # ---------- BOTTOM RIGHT ----------
            html.Div([
                html.H3("Bottom-right quadrant"),
                html.Div("Placeholder for plot 4...")
            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "box-sizing": "border-box"
            })
        ], style={
            "display": "flex", "flex-direction": "row", "width": "100%",
        }),
    ])
])


# Bump custom weights logic
@app.callback(
    Output("custom-weights", "style"),
    Input("metric", "value")
)
def toggle_custom_weights(metric):
    if metric == "Custom Reward":
        return {"display": "block"}
    return {"display": "none"}


# Bump Callback
@app.callback(
    Output("bumpchart", "figure"),
    Input("metric", "value"),
    Input("method", "value"),
    Input("w1", "value"),
    Input("w2", "value"),
    Input("w3", "value"),
)
def chart_callback(metric, method, w1, w2, w3):
    return update_bump(data_bump, color_map, metric, method, w1, w2, w3, n_top=10)


# State Space Callback
@app.callback(
    Output("state-space-plot", "figure"),
    Input("projection-method", "value"),
    Input("projection-param", "value")
)
def update_projection(method, param_value):
    return update_state_space(metadata_objects, features_objects, method=method, param_value=param_value)


# DAG Callback
@app.callback(
    [Output("dag-graph", "elements"),
     Output("dag-graph", "stylesheet"),
     Output("dag-graph", "layout"),
     Output("dag-legend", "figure"),
     Output("prev-node-truncation", "data")],
    Input("flow-attr", "value"),
    Input("trajectory-truncation", "value"),
    Input("edge-truncation", "value"),
    Input("dag-layout", "value"),
    State("prev-node-truncation", "data")
)
def update_dag_callback(flow_attr, trajectory_truncation, edge_truncation, layout_name, prev_node):
    global dag_data
    if trajectory_truncation != prev_node:
        dag_data = prepare_DAG(
            "data_trajectories.csv",
            n_trajectories=int(trajectory_truncation)
        )

    result = update_DAG(
        dag_data,
        flow_attr=flow_attr,
        truncation_pct=edge_truncation
    )

    # Configure layout based on selection
    layout_config = {
        'name': layout_name,
        'directed': True,
        'animate': False
    }

    # Add layout-specific parameters
    if layout_name == 'klay':
        layout_config['spacingFactor'] = 1.0
    elif layout_name == 'dagre':
        layout_config['spacingFactor'] = 1.0
        layout_config['rankDir'] = 'LR'  # Left to right
    elif layout_name == 'breadthfirst':
        layout_config['spacingFactor'] = 1.5
        layout_config['roots'] = '[id = "START"]'

    return result['elements'], result['stylesheet'], layout_config, result['legend'], trajectory_truncation


# Run the dashboard
if __name__ == "__main__":
    app.run(debug=True)