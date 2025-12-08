import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate

from plot_utils import *

# Load data
data = pd.read_csv('train_data.csv')
data.insert(loc=10, column="istestset", value=False)
data_dps = None
data_dpt = None
data_test = pd.read_csv('testset.csv')
data_test["final_id"]+=len(data)
data_test["iteration"]=0
data_test.insert(loc=10, column="istestset", value=True)
testset_reward_bounds = (data_test["total_reward"].min(), data_test["total_reward"].max())


# add ranked reward for filtering for performance
last_rewards = data.groupby('final_id')['total_reward'].transform('last')
ranks = last_rewards.rank(method='dense', ascending=False).astype(int)
data.insert(9, 'reward_ranked', ranks)




app = dash.Dash(__name__)

# Load extra layouts for cytoscape
cyto.load_extra_layouts()

flow_options = ["logprobs_forward", "logprobs_backward",
                "logprobs_forward_change", "logprobs_backward_change"]

app.layout = html.Div([
    dcc.Store(id="selected-objects", data=[]),  # selected objects based on final_id
    dcc.Store(id="data-dps", data=data_dps), #downprojections
    dcc.Store(id="data-dpt", data=data_dpt),

    # ================= LEFT COLUMN (12%) =================
    html.Div([

        html.H4("General"),

        html.Div([
            html.Button("Clear selection", id="clear-selection", n_clicks=0, style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),

            # -------- Iterations --------
            html.Div([
                html.Div("Iterations", style={"textAlign": "center"}),
                dcc.RangeSlider(
                    id="iteration",
                    min=0,
                    max=data["iteration"].max(),
                    step=25,
                    value=[0, data["iteration"].max()],
                    #marks={500: "500", 5000: "5000", 10000: "10000"},
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),

            # -------- Projection method --------
            html.Div([
                html.Div("Projection method", style={"textAlign": "center"}),
                dcc.Dropdown(
                    id="projection-method",
                    options=[
                        {"label": "UMAP", "value": "umap"},
                        {"label": "t-SNE", "value": "tsne"}
                    ],
                    value="tsne",
                    clearable=False
                )
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),

            # -------- Projection param --------
            html.Div([
                html.Div(
                    id="projection-param-label",
                    style={"textAlign": "center"}
                ),
                dcc.Slider(
                    id="projection-param",
                    min=5,
                    max=50,
                    step=1,
                    value=15,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),

            # -------- Limit Trajectories --------
            html.Div([
                html.Div("Limit trajectories", style={"textAlign": "center"}),
                html.Div(
                    "Keep only the top N trajectories for trajectory-visualizations. "
                    "(temporary for faster loading)",
                    style={"textAlign": "center", "font-size": "10px"}
                ),
                dcc.Slider(
                    id="limit-trajectories",
                    min=1,
                    max=100,
                    step=1,
                    value=5,
                    marks={0: '0', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),


            # -------- Limit Trajectories --------
            dcc.Checklist(["Use Testset"], [], id="use-testset", style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            })

        ], style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px"
        }),

        html.Div(style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px",
            "height": "40px"
        }),

        #---------------DAG Controls---------------

        html.H4("DAG"),

        html.Div([
            html.Div([

                # -------- Layout --------
                html.Div([
                    html.Div("Layout", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="dag-layout",
                        options=[
                            {"label": "Klay", "value": "klay"},
                            {"label": "Dagre", "value": "dagre"},
                            {"label": "Breadthfirst", "value": "breadthfirst"}
                        ],
                        value="breadthfirst",
                        clearable=False
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px"
                }),

                # -------- Edge coloring --------
                html.Div([
                    html.Div("Edge coloring", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="flow-attr",
                        options=[{"label": f, "value": f} for f in flow_options],
                        value="logprobs_forward",
                        clearable=False
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px"
                }),

                # -------- Truncate Edges --------
                html.Div([
                    html.Div("Truncate Edges", style={"textAlign": "center"}),
                    dcc.Slider(
                        id="edge-truncation",
                        min=0,
                        max=100,
                        step=5,
                        value=100,
                        marks={0: '0%', 50: '50%', 100: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px"
                }),

            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "50px"
            })

        ], style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px"
        })

    ], style={
        "width": "12%",
        "minWidth": "180px",
        "padding": "12px",
        "height": "100vh",
        "borderRight": "1px solid #ddd",
        "overflow": "auto"
    })
    ,



    # ================= RIGHT COLUMN (88%) =================
    html.Div([

        dcc.Store(id="prev-node-truncation", data=0),

        # ================= TOP ROW =================
        html.Div([

            # ---------- TOP LEFT ----------
            html.Div([
                html.Div(
                    dcc.Graph(id="bumpchart", clear_on_unhover=True),
                    style={"height": "100%", "width": "100%"}
                ),
                dcc.Tooltip(id="image-tooltip3", direction="left"),

            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "boxSizing": "border-box",
                "overflow": "hidden"
            }),


            # ---------- TOP RIGHT ----------
            html.Div([

                html.Div(
                    dcc.Graph(id="state-space-plot", clear_on_unhover=True),
                    style={"height": "100%", "width": "100%"}
                ),

                dcc.Tooltip(id="image-tooltip1"),

            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "boxSizing": "border-box",
                "overflow": "hidden"
            }),

        ], style={
            "display": "flex",
            "flexDirection": "row",
            "width": "100%"
        }),


        # ================= BOTTOM ROW =================
        html.Div([

            # ---------- BOTTOM LEFT (DAG) ----------
            html.Div([

                html.Div([

                    cyto.Cytoscape(
                        id='dag-graph',
                        layout={
                            'name': 'klay',
                            'directed': True,
                            'spacingFactor': 0.5,
                            'animate': False
                        },
                        style={'flex': '1', 'height': '49vh', 'width': '0px', 'background-color': '#cfcfcf'},
                        elements=[],
                        stylesheet=[]
                    ),

                    dcc.Graph(
                        id='dag-legend',
                        style={'width': '75px', 'height': '49vh', 'flex': '0 0 75px'}
                    )

                ], style={
                    "display": "flex",
                    "flexDirection": "row",
                    "width": "100%"
                })

            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "boxSizing": "border-box",
                "overflow": "hidden"
            }),


            # ---------- BOTTOM RIGHT (TRAJECTORY VISUALIZATION) ----------
            html.Div([

                html.Div(
                    dcc.Graph(id="trajectory-plot", clear_on_unhover=True),
                    style={"height": "100%", "width": "100%"}
                ),

                dcc.Tooltip(id="image-tooltip2"),

            ], style={
                "flex": 1,
                "border": "1px solid #ddd",
                "padding": "5px",
                "height": "49vh",
                "boxSizing": "border-box",
                "overflow": "hidden"
            }),

        ], style={
            "display": "flex",
            "flexDirection": "row",
            "width": "100%"
        })

    ], style={
        "width": "88%",
        "height": "100vh",
        "overflow": "hidden"
    })

], style={
    "display": "flex",
    "flexDirection": "row",
    "width": "100vw",
    "height": "100vh"
})


# Downprojection parameter header
@app.callback(
    Output("projection-param-label", "children"),
    Output("projection-param", "min"),
    Output("projection-param", "max"),
    Output("projection-param", "step"),
    Output("projection-param", "marks"),
    Output("projection-param", "value"),
    Input("projection-method", "value")
)
def update_projection_param(method):

    if method == "umap":
        # UMAP wants n_neighbors (usually 2–200)
        return "n_neighbors", 2, 200, 1, {2: "2", 50: "50", 100: "100", 200: "200"}, 15

    else:
        # t-SNE wants perplexity (usually 5–50)
        return "perplexity", 5, 50, 1, {5: "5", 25: "25", 50: "50"}, 30


# Main selection update
@app.callback(
    Output("selected-objects", "data"),
    Input("clear-selection", "n_clicks"),
    Input("state-space-plot", "selectedData"),
    Input("trajectory-plot", "selectedData"),
    Input("bumpchart", "selectedData"),
    Input("dag-graph", "tapNodeData"),
    State("selected-objects", "data"),
    prevent_initial_call=True
)
def update_selected_objects(clear_clicks, ss_select, traj_select, bump_select, dag_node, current_ids):

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    trigger = ctx.triggered[0]["prop_id"]

    # -------- Clear button --------
    if "clear-selection" in trigger:
        return []

    # ---------- State-space lasso ----------
    if "state-space-plot.selectedData" in trigger:
        if not ss_select or not ss_select.get("points"):
            return no_update

        selected_ids = {pt["customdata"][0] for pt in ss_select["points"]}
        return list(selected_ids)

    # ---------- Trajectory lasso ----------
    elif "trajectory-plot.selectedData" in trigger:
        if not traj_select or not traj_select.get("points"):
            return no_update

        selected_ids = {pt["customdata"][1] for pt in traj_select["points"]}
        return list(selected_ids)

    # ---------- Bump chart lasso ----------
    elif "bumpchart.selectedData" in trigger:
        if not bump_select or not bump_select.get("points"):
            return no_update

        selected_ids = {pt["customdata"][0] for pt in bump_select["points"]}
        return list(selected_ids)

    # ---------- DAG node click ----------
    elif "dag-graph.tapNodeData" in trigger:
        if not dag_node:
            return no_update

        text = dag_node.get("id")
        final_id = data.loc[data["text"] == text, "final_id"]\
                       .dropna()\
                       .unique()\
                       .tolist()

        return final_id if final_id else []

    return no_update


# Compute downprojections
@app.callback(
    Output("data-dps", "data"),
    Output("data-dpt", "data"),
    Input("projection-method", "value"),
    Input("projection-param", "value"),
    Input("limit-trajectories", "value"),
    Input(  "iteration", "value"),
    Input(  "use-testset", "value")
)
def compute_downprojections(method, param_value, trajectories, iteration, use_testset):
    cols_to = 12
    objs = data[data["iteration"] <= iteration[1]]
    objs = objs[objs["iteration"] >= iteration[0]]
    if use_testset:
        objs = pd.concat((objs, data_test))
    objs = objs[objs["features_valid"] == True]

    #states
    data_s = objs[(objs["final_object"] == True) | (objs["istestset"]==True)].copy()
    metadata_s = data_s.iloc[:, :cols_to].reset_index(drop=True)
    features_s = data_s.iloc[:, cols_to:].reset_index(drop=True)
    print(len(data_s), data_s.columns[:cols_to+1])

    #trajectories
    top_ranks = sorted(objs['reward_ranked'].dropna().unique())[:trajectories]
    data_t = objs[objs["reward_ranked"].isin(top_ranks) | (objs["istestset"] == True)]
    metadata_t = data_t.iloc[:, :cols_to].reset_index(drop=True)
    features_t = data_t.iloc[:, cols_to:].reset_index(drop=True)

    # Downprojection
    if method == "tsne":
        proj_s = manifold.TSNE(
            perplexity=min(param_value, len(features_s)-1),
            init='pca',
            learning_rate='auto'
        ).fit_transform(features_s)
        proj_t = manifold.TSNE(
            perplexity=min(param_value, len(features_t)-1),
            init='pca',
            learning_rate='auto'
        ).fit_transform(features_t)
    elif method == "umap":
        reducer = UMAP(n_neighbors=param_value)
        proj_s = reducer.fit_transform(features_s)
        proj_t = reducer.fit_transform(features_t)
    else:
        raise NotImplementedError("Method not implemented")

    data_s = pd.concat([metadata_s, pd.DataFrame(proj_s, columns=['X', 'Y'])], axis=1)
    data_t = pd.concat([metadata_t, pd.DataFrame(proj_t, columns=['X', 'Y'])], axis=1)
    print(len(data_s), len(data_t), "changed")
    return data_s.to_dict("records"), data_t.to_dict("records")

# Bump Callback
@app.callback(
    Output("bumpchart", "figure"),
    Input("iteration", "value"),
    Input("selected-objects", "data"),
    Input("use-testset", "value"),
)
def bump_callback(iteration, selected_ids, use_testset):
    tmp = data[data["final_object"] == True]
    tmp = tmp.iloc[:, :10]
    tmp = tmp[tmp["iteration"] <= iteration[1]]
    tmp = tmp[tmp["iteration"] >= iteration[0]]
    tmp["istestset"]=False
    if use_testset:
        return update_bump(tmp, 30, selected_ids, testset_reward_bounds)

    return update_bump(tmp, 30, selected_ids, None)


# State Space Callback
@app.callback(
    Output("state-space-plot", "figure"),
    Output("trajectory-plot", "figure"),
    Input("selected-objects", "data"),
    Input("data-dps", "data"),
    Input("data-dpt", "data")
)
def update_projection_plots(selected_ids, data_s, data_t):
    return(
        update_state_space(pd.DataFrame(data_s), selected_ids),
        update_state_space_t(pd.DataFrame(data_t), selected_ids)
    )

# DAG Callback
@app.callback(
    [Output("dag-graph", "elements"),
     Output("dag-graph", "stylesheet"),
     Output("dag-graph", "layout"),
     Output("dag-legend", "figure")],
    Input("flow-attr", "value"),
    Input("edge-truncation", "value"),
    Input("dag-layout", "value"),
    Input("limit-trajectories", "value"),
    Input(  "iteration", "value"),
    Input("selected-objects", "data"),
)
def update_dag_callback(flow_attr, edge_truncation, layout_name, trajectories, iteration, selected_ids):
    tmp = data.iloc[:, :10]
    tmp = tmp[tmp["iteration"] <= iteration[1]]
    tmp = tmp[tmp["iteration"] >= iteration[0]]
    top_ranks = sorted(tmp['reward_ranked'].dropna().unique())[:trajectories]
    tmp = tmp[tmp["reward_ranked"].isin(top_ranks)]
    if selected_ids:
        tmp=tmp[tmp["final_id"].isin(selected_ids)]
    graph = prepare_graph(tmp)
    #selected_texts = list(set(tmp["text"])) if selected_ids else []
    #print(selected_texts)
    #print(len(selected_texts))

    result = update_DAG(
        graph,
        flow_attr,
        edge_truncation,
        selected_ids
    )

    # Configure layout based on selection
    layout_config = {
        'name': layout_name,
        'directed': True,
        'animate': False
    }

    # Add layout-specific parameters
    if layout_name == 'klay':
        layout_config['spacingFactor'] = 1
    elif layout_name == 'dagre':
        layout_config['spacingFactor'] = 0.7
        layout_config['rankDir'] = 'LR'  # Left to right
    elif layout_name == 'breadthfirst':
        layout_config['spacingFactor'] = 1
        layout_config['roots'] = '[id = "START"]'

    if not selected_ids:
        layout_config['spacingFactor'] *= 0.7

    print(layout_config['spacingFactor'])

    return result['elements'], result['stylesheet'], layout_config, result['legend']

#hover state space
@app.callback(
    Output("image-tooltip1", "show"),
    Output("image-tooltip1", "bbox"),
    Output("image-tooltip1", "children"),
    Input("state-space-plot", "hoverData"),
)
def display_image_tooltip1(hoverData):
    if hoverData is None:
        return False, None, None

    # Extract bounding box for positioning
    bbox = hoverData["points"][0]["bbox"]

    # Extract base64 image saved in customdata
    _, iteration, reward, image_b64 = hoverData["points"][0]["customdata"]

    # Build HTML content
    children = [
        html.Div([
            html.Img(
                src=f"data:image/svg+xml;base64,{image_b64}",
                style={"width": "150px", "height": "150px"}
            ),
            html.Div(f"Iteration: {iteration}"),
            html.Div(f"Reward: {reward:.3f}"),
        ])
    ]

    return True, bbox, children

#hover state space trajectories
@app.callback(
    Output("image-tooltip2", "show"),
    Output("image-tooltip2", "bbox"),
    Output("image-tooltip2", "children"),
    Input("trajectory-plot", "hoverData"),
)
def display_image_tooltip2(hoverData):
    if hoverData is None:
        return False, None, None

    # Extract bounding box for positioning
    bbox = hoverData["points"][0]["bbox"]

    # Extract base64 image saved in customdata
    image_b64 = hoverData["points"][0]["customdata"][0]

    # Build HTML content
    children = [
        html.Div([
            html.Img(
                src=f"data:image/svg+xml;base64,{image_b64}",
                style={"width": "150px", "height": "150px"}
            ),
        ])
    ]

    return True, bbox, children


#hover bump plot
@app.callback(
    Output("image-tooltip3", "show"),
    Output("image-tooltip3", "bbox"),
    Output("image-tooltip3", "children"),
    Input("bumpchart", "hoverData"),
)
def display_image_tooltip3(hoverData):
    if hoverData is None:
        return False, None, None

    point = hoverData["points"][0]

    # Check if this point has customdata (skip shading area)
    if "customdata" not in point or point["customdata"] is None:
        return False, None, None

    # Extract bounding box for positioning
    bbox = point["bbox"]

    # Extract base64 image saved in customdata
    _, value, image_b64, reward = point["customdata"]

    # Build HTML content
    children = [
        html.Div([
            html.Img(
                src=f"data:image/svg+xml;base64,{image_b64}",
                style={"width": "150px", "height": "150px"}
            ),
            html.Div(f"Rank: {value}"),
            html.Div(f"Reward: {reward:.4f}"),
        ])
    ]

    return True, bbox, children

# Run the dashboard
if __name__ == "__main__":
    app.run(debug=True)