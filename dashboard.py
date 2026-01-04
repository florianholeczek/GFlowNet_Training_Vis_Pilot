import sqlite3
import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table, ctx
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
from sklearn import manifold
from umap import UMAP
from plot_utils import *


# Load data
data = pd.read_csv('traindata1/train_data.csv')
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], assets_folder="traindata1")

# Load extra layouts for cytoscape
cyto.load_extra_layouts()

app.layout = html.Div([
    dcc.Store(id="selected-objects", data=[]),
    dcc.Store(id="data-dps", data=data_dps),
    dcc.Store(id="data-dpt", data=data_dpt),
    dcc.Store(id="build-ids", data= ["#"]),
    dcc.Store(id="max-frequency", data= 0),
    dcc.Store(id="dag-overview-tid-list", data= []),
    dcc.Store(id="dag-overview-edge-list", data= []),

    # ================= LEFT SIDEBAR (12%) =================
    html.Div([

# -------- TAB SELECTOR --------
        html.H4("View"),
        html.Div([
            html.Button(
                "State Space",
                id="tab-state-space",
                n_clicks=0,
            ),
            html.Button(
                "DAG",
                id="tab-dag-view",
                n_clicks=0,
            )
        ], style={
            "display": "flex",
            "flexDirection": "column",
            #"gap": "12px"
        }),
        dcc.Store(id="active-tab", data="dag-view"),

        html.Div(style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px",
            "height": "40px"
        }),

        html.H4("General"),

        html.Div([
            html.Button("Clear selection", id="clear-selection", n_clicks=0, style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px",
                "border-radius": "8px",
                "border": "2px solid #e5e7eb",
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
                    tooltip={"placement": "bottom", "always_visible": False}
                ),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "6px"
            }),

            # -------- Use Testset --------
            dcc.Checklist(["Use Testset"], [], id="use-testset", style={
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

        html.H4("Projection", id="sidebar-tab-header"),
        # --------------- Projection Controls ---------------


        html.Div([
            html.Div([
                # -------- Projection method --------
                html.Div([
                    html.Div("Method", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="projection-method",
                        options=[
                            {"label": "UMAP", "value": "umap"},
                            {"label": "t-SNE", "value": "tsne"}
                        ],
                        value="tsne",
                        clearable=False,
                        style={"color": "black"}
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px",
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

            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "50px"
            })

        ], id = "sidebar-projection", style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px"
        }),


        # --------------- DAG Controls ---------------
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
                        value="klay",
                        clearable=False,
                        style={"color": "black"}
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px"
                }),

                # -------- Metric --------
                html.Div([
                    html.Div("Metric", style={"textAlign": "center"}),
                    dcc.Dropdown(
                        id="dag-metric",
                        options=["highest", "lowest", "variance", "frequency"],
                        value="highest",
                        clearable=False,
                        style={"color": "black"}
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "6px"
                }),

                # -------- Direction --------
                html.Div([
                    html.Div("Direction", style={"textAlign": "center"}),
                    dcc.RadioItems(
                        id="dag-direction",
                        options=["forward", "backward"],
                        value="forward",
                    ),
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

        ], id = "sidebar-dag", style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "50px"
        })

    ], style={
        "width": "12%",
        "minWidth": "180px",
        "maxWidth": "250px",
        "padding": "12px",
        "height": "100vh",
        #"borderRight": "1px solid #ddd",
        "overflow": "auto"
    }),

    # ================= RIGHT CONTENT AREA (88%) =================
    html.Div([

        # ================= STATE-SPACE TAB =================
        html.Div([
            # TOP ROW
            html.Div([
                # TOP LEFT
                html.Div([
                    html.Div(
                        dcc.Graph(id="bumpchart", clear_on_unhover=True),
                        style={"height": "100%", "width": "100%"}
                    ),
                    dcc.Tooltip(id="image-tooltip3", direction="left"),
                ], style={
                    "flex": 1,
                    #"border": "1px solid #ddd",
                    "padding": "5px",
                    "height": "49vh",
                    "boxSizing": "border-box",
                    "overflow": "hidden"
                }),

                # TOP RIGHT
                html.Div([
                    html.Div(
                        dcc.Graph(id="state-space-plot", clear_on_unhover=True),
                        style={"height": "100%", "width": "100%"}
                    ),
                    dcc.Tooltip(id="image-tooltip1"),
                ], style={
                    "flex": 1,
                    #"border": "1px solid #ddd",
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

            # BOTTOM ROW
            html.Div([
                # BOTTOM LEFT - EMPTY
                html.Div([], style={
                    "flex": 1,
                    #"border": "1px solid #ddd",
                    "padding": "5px",
                    "height": "49vh",
                    "boxSizing": "border-box",
                    "overflow": "hidden"
                }),

                # BOTTOM RIGHT - TRAJECTORY PLOT
                html.Div([
                    html.Div(
                        dcc.Graph(id="trajectory-plot", clear_on_unhover=True),
                        style={"height": "100%", "width": "100%"}
                    ),
                    dcc.Tooltip(id="image-tooltip2"),
                ], style={
                    "flex": 1,
                    #"border": "1px solid #ddd",
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
        ], id="state-space-tab", style={"display": "block"}),

        # ================= DAG TAB =================
        html.Div([
            html.Div([
                # LEFT SIDE - DAG AREA
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id="dag-overview",
                            clear_on_unhover=True,
                            style={"height": "100%", "width": "100%"},
                            config={"responsive": True},
                        ),
                    ], style={
                        "height": "24vh",
                        #"border": "1px solid #ddd",
                        "boxSizing": "border-box"
                    }),

                    html.Div("DAG-title", id="dag-title", style={
                        "height": "20px",
                        #"border": "1px solid #ddd",
                        "boxSizing": "border-box",
                        "padding-top": "3px",
                        "font-size": "12px",
                        "font-weight": "bold",
                        "margin-top": "2px",
                    }),
                    html.Div("DAG-subtitle", id="dag-subtitle", style={
                        "height": "24px",
                        #"border": "1px solid #ddd",
                        "boxSizing": "border-box",
                        "padding-top": "3px",
                        "font-size": "10px",
                        "margin-bottom": "13px",
                        "margin-top": "2px",
                        "whiteSpace": "pre-line",
                    }),

                    html.Div([
                        cyto.Cytoscape(
                            id='dag-graph',
                            layout={
                                'name': 'klay',
                                'directed': True,
                                'spacingFactor': 0.5,
                                'animate': False
                            },
                            style={'flex': '1', 'height': '100%', 'width': '0px', 'background-color': '#222222'},
                            elements=[],
                            stylesheet=[]
                        ),
                    ], style={
                        "display": "flex",
                        "flexDirection": "row",
                        "flex": 1,
                        #"border": "1px solid #ddd",
                        "boxSizing": "border-box"
                    })
                ], style={
                    "flex": 1,
                    "display": "flex",
                    "flexDirection": "column",
                    "height": "100vh"
                }),

                # RIGHT SIDE - DATA TABLE
                dash_table.DataTable(
                    id='dag-table',
                    columns=[
                        {
                            "name": "Image",
                            "id": "image",
                            "presentation": "markdown",
                        },
                        {
                            "name": "Final",
                            "id": "node_type",
                            "type": "any",
                        },
                        {
                            "name": "Metric",
                            "id": "logprobs",
                            "type": "numeric",
                            "format": Format(precision=4, scheme=Scheme.fixed),
                        },
                        {
                            "name": "Reward",
                            "id": "reward",
                            "type": "numeric",
                            "format": Format(precision=4, scheme=Scheme.fixed),
                        },
                    ],
                    row_selectable="multi",
                    filter_action="native",
                    sort_action="native",
                    selected_row_ids=[],
                    page_size=10,
                    markdown_options={"html": True},
                    style_cell={
                        'fontFamily': 'Arial',
                        'backgroundColor': '#222222',
                    },
                    style_header={
                        'backgroundColor': '#222222',
                        'fontWeight': 'bold'
                    },
                    style_table={'width': '500px', 'height': '95vh', 'flex': '0 0 400px', 'overflow': 'auto'}
                )
            ], style={
                "display": "flex",
                "flexDirection": "row",
                "height": "98vh"
            })
        ], id="dag-tab", style={"display": "none"})

    ], style={
        "width": "88%",
        "height": "100vh",
        "overflow": "hidden"
    }),
    dcc.Tooltip(
        id="image-tooltip4",
        direction="bottom",
        style = {"zIndex": 999, "pointerEvents": "none", "overflow": "visible"}
    ),

], style={
    "display": "flex",
    "flexDirection": "row",
    "width": "100vw",
    "height": "100vh"
})


# ================= CALLBACK TO SWITCH TABS =================
@app.callback(
    [Output("state-space-tab", "style"),
     Output("dag-tab", "style"),
     Output("active-tab", "data"),
     Output("tab-state-space", "style"),
     Output("tab-dag-view", "style"),
     Output("sidebar-projection", "style"),
     Output("sidebar-dag", "style"),
     Output("sidebar-tab-header", "children")],
    [Input("tab-state-space", "n_clicks"),
     Input("tab-dag-view", "n_clicks")],
    [State("active-tab", "data")]
)
def switch_tabs(state_clicks, dag_clicks, current_tab):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = "tab-dag-view"
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "tab-state-space":
        active_tab = "state-space"
    elif button_id == "tab-dag-view":
        active_tab = "dag-view"
    else:
        active_tab = current_tab

    # Base button styles
    active_style = {
        "border": "2px solid #3b82f6",
        "backgroundColor": "#3b82f6",
        "color": "white",
        "transition": "all 0.3s ease",
    }
    inactive_style = {
        "border": "2px solid #e5e7eb",
        "backgroundColor": "white",
        "color": "#6b7280",
        "transition": "all 0.3s ease",
    }
    top_style = {"border-top-left-radius": "8px", "border-top-right-radius": "8px"}
    bottom_style = {"border-bottom-left-radius": "8px", "border-bottom-right-radius": "8px"}

    # Apply styles based on active tab
    if active_tab == "state-space":

        return (
            {"display": "block"},
            {"display": "none"},
            "state-space",
            active_style | top_style,
            inactive_style | bottom_style,
            {"display": "block"},
            {"display": "none"},
            "Projection"
        )
    else:
        return (
            {"display": "none"},
            {"display": "block"},
            "dag-view",
            inactive_style | top_style,
            active_style | bottom_style,
            {"display": "none"},
            {"display": "block"},
            "DAG"
        )

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
        return "n_neighbors", 2, 200, 1, {2: "2", 50: "50", 100: "100", 200: "200"}, 15
    else:
        return "perplexity", 5, 50, 1, {5: "5", 25: "25", 50: "50"}, 30

Input("dag-overview", "selectedData"),

# Main selection update
@app.callback(
    Output("selected-objects", "data"),
    Output("dag-table", "data"),
    Output("dag-table", "selected_rows"),
    Input("clear-selection", "n_clicks"),
    Input("state-space-plot", "selectedData"),
    Input("trajectory-plot", "selectedData"),
    Input("bumpchart", "selectedData"),
    Input("dag-graph", "tapNodeData"),
    Input("dag-overview", "selectedData"),
    State("selected-objects", "data"),
    State("build-ids", "data"),
    State("dag-overview-tid-list", "data"),
    prevent_initial_call=True
)
def update_selected_objects(clear_clicks, ss_select, traj_select, bump_select, dag_node, selected_tids, current_ids, build_ids, tid_list):

    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    trigger = ctx.triggered[0]["prop_id"]

    # -------- Clear button --------
    if "clear-selection" in trigger:
        return [], None, []

    # ---------- State-space lasso ----------
    if "state-space-plot.selectedData" in trigger:
        if not ss_select or not ss_select.get("points"):
            return no_update

        selected_ids = {pt["customdata"][0] for pt in ss_select["points"]}
        return list(selected_ids), None, []

    # ---------- Trajectory lasso ----------
    elif "trajectory-plot.selectedData" in trigger:
        if not traj_select or not traj_select.get("points"):
            return no_update, None, []

        selected_ids = {pt["customdata"][1] for pt in traj_select["points"]}
        return list(selected_ids), None, []

    # ---------- Bump chart lasso ----------
    elif "bumpchart.selectedData" in trigger:
        if not bump_select or not bump_select.get("points"):
            return no_update, None, []

        selected_ids = {pt["customdata"][0] for pt in bump_select["points"]}
        return list(selected_ids), None, []

    # ---------- DAG node click ----------
    elif "dag-graph.tapNodeData" in trigger:
        if not dag_node:
            return no_update
        if dag_node.get("id") == "#":
            # root selection clears selection
            return [], None, []


        iteration0 = int(dag_node.get("iteration0"))
        iteration1 = int(dag_node.get("iteration1"))

        if dag_node.get("node_type") == 'handler':
            text = dag_node.get("id")[8:]
            conn = sqlite3.connect("traindata1/traindata1_1.db")

            if dag_node.get("metric") in ["highest", "lowest", "variance"]:
                col = "logprobs_" + dag_node.get("direction")
                query = f"""
                    WITH ranked AS (
                        SELECT
                            target,
                            iteration,
                            {col} AS logprobs,
                            AVG({col}) OVER (PARTITION BY target) AS avg_logprobs,
                            ROW_NUMBER() OVER (
                                PARTITION BY target
                                ORDER BY iteration DESC
                            ) AS rn
                        FROM edges
                        WHERE source = ?
                        AND iteration BETWEEN ? AND ?
                    )
                    SELECT
                        target,
                        logprobs AS latest_logprobs,
                        avg_logprobs
                    FROM ranked
                    WHERE rn = 1;
                    """
            else: #frequency
                query = f"""
                    SELECT
                        target,
                        COUNT(*) AS latest_logprobs
                    FROM edges
                    WHERE source = ?
                      AND iteration BETWEEN ? AND ?
                    GROUP BY target;
                    """
            children_e = pd.read_sql_query(query, conn, params=[text, iteration0, iteration1])

            if dag_node.get("metric") == "variance":
                children_e["latest_logprobs"] -= children_e["avg_logprobs"]
            children_e.rename(columns={"target": "id", "latest_logprobs": "logprobs"}, inplace=True)
            children_e = children_e[["id", "logprobs"]]

            targets = list(children_e["id"])
            placeholders = ",".join("?" for _ in targets)
            query = f"""
                        SELECT DISTINCT
                            id, image, node_type, reward
                        FROM nodes
                        WHERE id IN ({placeholders})
                    """
            children_n = pd.read_sql_query(query, conn, params=targets)
            conn.close()
            children = pd.merge(children_n, children_e, on="id")
            children["image"] = children["image"].apply(
                lambda p: f"![img]({p.replace('traindata1', 'assets')})"
            )
            children["node_type"] = children["node_type"].eq("final")
            selected_row_ids = list(set.intersection(set(build_ids), set(list(children["id"]))))
            selected_rows = [
                idx for idx, row in enumerate(children.to_dict("records"))
                if row["id"] in selected_row_ids
            ]
            return no_update, children.to_dict("records"), selected_rows
        else:
            text = dag_node.get("id")
            conn = sqlite3.connect("traindata1/traindata1_1.db")
            query = f"""
                        SELECT DISTINCT
                            trajectory_id
                        FROM edges
                        WHERE target = ?
                          AND iteration BETWEEN ? AND ?
                    """
            selected_ids = pd.read_sql_query(query, conn, params=[text, iteration0, iteration1])
            selected_ids = list(selected_ids["trajectory_id"])
            conn.close()
            return (selected_ids, None, []) if selected_ids else ([], None, [])

    elif "dag-overview.selectedData" in trigger:
        if not selected_tids or "range" not in selected_tids:
            return no_update

        x_range = np.round(selected_tids["range"]["x"]).astype(int)
        t_ids = tid_list[x_range[0]:x_range[1] + 1]
        t_ids = list({elem for sublist in t_ids for elem in sublist})
        iterations = np.round(selected_tids["range"]["y"]).astype(int).tolist()

        # get trajectory ids that are also in iteration range and then all node ids for these trajectory_ids
        conn = sqlite3.connect("traindata1/traindata1_1.db")
        placeholders = ",".join("?" for _ in t_ids)
        query = f"""
        SELECT DISTINCT trajectory_id
        FROM edges
        WHERE trajectory_id IN ({placeholders})
            AND iteration BETWEEN ? AND ?
        """
        params = t_ids + [iterations[0], iterations[1]]
        selected_ids = pd.read_sql_query(query, conn, params=params)
        selected_ids = list(set(selected_ids["trajectory_id"].to_list())) + ["#"]
        print(selected_ids)
        return selected_ids, None, []


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
        reducer_s = UMAP(n_neighbors=min(param_value, len(features_s)-1))
        reducer_t = UMAP(n_neighbors=min(param_value, len(features_t)-1))
        proj_s = reducer_s.fit_transform(features_s)
        proj_t = reducer_t.fit_transform(features_t)
    else:
        raise NotImplementedError("Method not implemented")

    data_s = pd.concat([metadata_s, pd.DataFrame(proj_s, columns=['X', 'Y'])], axis=1)
    data_t = pd.concat([metadata_t, pd.DataFrame(proj_t, columns=['X', 'Y'])], axis=1)
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
     Output("dag-title", "children"),
     Output("dag-subtitle", "children")],
    Input("dag-layout", "value"),
    Input("dag-direction", "value"),
    Input("dag-metric", "value"),
    Input(  "iteration", "value"),
    Input("selected-objects", "data"),
    Input("build-ids", "data"),
    Input("max-frequency", "data")
)
def update_dag(layout_name, direction, metric, iteration, selected_objects, build_ids, max_freq):
    add_handlers = True
    if selected_objects:
        print("selected objects update")
        # If final objects are selected via another vis, display the full dag of these
        conn = sqlite3.connect("traindata1/traindata1_1.db")
        placeholders = ",".join("?" for _ in selected_objects)
        query = f"""
                    SELECT DISTINCT
                        target
                    FROM edges
                    WHERE trajectory_id IN ({placeholders})
                      AND iteration BETWEEN ? AND ?
                """
        build_ids = pd.read_sql_query(query, conn, params=selected_objects +  [iteration[0], iteration[1]])
        build_ids = list(build_ids['target']) + ['#']
        conn.close()
        add_handlers = False
    elif not build_ids:
        build_ids = ["#"]

    result = update_DAG(
        iteration,
        direction,
        metric,
        max_freq,
        add_handlers,
        build_ids=build_ids,
    )

    # Configure layout based on selection
    layout_config = {
        'name': layout_name,
        'directed': True,
        'animate': False
    }

    # Add layout-specific parameters
    if layout_name == 'klay':
        layout_config['spacingFactor'] = 1.2
    elif layout_name == 'dagre':
        layout_config['spacingFactor'] = 1
        layout_config['rankDir'] = 'LR'  # Left to right
    elif layout_name == 'breadthfirst':
        layout_config['spacingFactor'] = 1.2
        layout_config['roots'] = '[id = "START"]'

    if add_handlers:
        title = "Directed Acyclic Graph, Mode: Expand"
        subtitle = "Click on 'Select children' nodes to expand the Graph and click on the root to collapse it. Select a node or items from other visuals to switch to selection mode. Edge coloring: "
    else:
        title = "Directed Acyclic Graph, Mode: Selection"
        subtitle = "Shows all trajectories going through the selected items. Clear selection or select the root to switch to expanding mode. Edge coloring: "
    if metric in ["highest", "lowest"]:
        subtitle += f"{metric.capitalize()} {direction} logprobabilities of the edge over selected iterations."
    elif metric == "variance":
        subtitle += f"Latest {direction} logprobability of the edge (in selected iterations) - mean ({direction} logprobabilities) over selected iterations."
    elif metric == "frequency":
        subtitle += "Frequency of the edge over selected iterations."
    return result['elements'], result['stylesheet'], layout_config, title, subtitle

# Callback for dag-table
@app.callback(
    Output('build-ids', 'data'),
    Input('dag-table', 'selected_row_ids'),
    State('dag-table', 'data'),
    State("build-ids", "data")
)
def save_selected_rows(selected_rows, table_data, build_ids):
    print(selected_rows)
    if selected_rows:
        children = set([r["id"] for r in table_data])
        unselected = children - set(selected_rows)
        build_ids = set(build_ids) - unselected
        build_ids = set(selected_rows) | build_ids | set(["#"])
        return list(build_ids)
    elif table_data:
        children = set([r["id"] for r in table_data])
        return list(set(build_ids)-children)
    else:
        return ["#"]

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
    print(hoverData)

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


#dag overview
@app.callback(
    Output("dag-overview", "figure"),
    Output("max-frequency", "data"),
    Output("dag-overview-tid-list", "data"),
    Output("dag-overview-edge-list", "data"),
    Input("dag-direction", "value"),
    Input("dag-metric", "value"),
    Input("iteration", "value"),
)
def update_dag_overview(direction, metric, iteration):
    fig, max_freq, ids, edge_list = update_DAG_overview(direction, metric, iteration)
    if max_freq:
        return fig, max_freq, ids, edge_list
    return fig, no_update, ids, edge_list

@app.callback(
    Output("image-tooltip4", "show"),
    Output("image-tooltip4", "bbox"),
    Output("image-tooltip4", "children"),
    Input("dag-overview", "hoverData"),
    State("dag-overview-edge-list", "data")
)
def display_image_tooltip4(hoverData, edge_list):
    if hoverData is None or hoverData["points"][0]["z"] is None:
        return False, None, None
    print(hoverData)

    value = hoverData["points"][0]["z"]
    bbox = hoverData["points"][0]["bbox"]
    #bbox["x0"] += 175
    #bbox["x1"] += 175
    idx = hoverData["points"][0]["x"]
    source, target = edge_list[idx]
    print(source, target, value)

    # get images
    conn = sqlite3.connect("traindata1/traindata1_1.db")
    query = f"SELECT image FROM nodes WHERE id=?"
    source_img = pd.read_sql_query(query, conn, params=[source])["image"][0]
    target_img = pd.read_sql_query(query, conn, params=[target])["image"][0]
    print(source_img, target_img)
    query = "SELECT DISTINCT iteration, logprobs_forward, logprobs_backward FROM edges WHERE source = ? AND target = ?"
    edge_data = pd.read_sql_query(query, conn, params=[source, target])
    print(edge_data)
    conn.close()

    def make_img(img_url):
        if img_url is None:
            return html.Div("root")
        return html.Img(src=img_url.replace('traindata1', 'assets'), style={"height": "100px"})

    children = html.Div(
        [
            html.Div(f"Value: {value}", style={"fontWeight": "bold"}),
            html.Div(
                [
                    html.Div(
                        [html.Div("Source:"), make_img(source_img)],
                        style={
                            "marginTop": "5px",
                            "marginRight": "50px",
                            "display": "flex",
                            "textAlign": "center",
                            "alignItems": "center",
                            "flexDirection": "column",
                            "minHeight": "80px",
                            "justifyContent": "flex-start",
                        }
                    ),
                    html.Div(
                        [html.Div("Target:"), make_img(target_img)],
                        style={"marginTop": "5px"}
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginTop": "5px",
                    "alignItems": "flex-start",
                },
            ),
            dcc.Graph(
                figure=edge_hover_fig(edge_data),
                config={"displayModeBar": False},
                style={"marginTop": "10px"}
            ),
        ],
        style={
            "color": "black",
            "backgroundColor": "white",
            "padding": "10px",
            "height": "auto",
            "width": "260px",
        },
    )

    return True, bbox, children


# Run the dashboard
if __name__ == "__main__":
    app.run(debug=True)