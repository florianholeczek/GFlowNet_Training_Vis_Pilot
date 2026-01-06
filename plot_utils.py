import base64
import sqlite3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def update_state_space_t(df, selected_ids=[]):
    """
    Create a trajectory visualization in state space with dimensionality reduction.

    Parameters:
    -----------
    data : pd.DataFrame
        The trajectory data with metadata in first 9 columns
    method : str
        'umap' or 'tsne'
    param_value : int
        n_neighbors for UMAP or perplexity for t-SNE
    n_trajectories : int
        Number of trajectories to visualize

    Returns:
    --------
    plotly.graph_objects.Figure
    """

    # Create figure
    fig = go.Figure()

    # Get unique trajectory IDs and their iterations
    unique_ids = df['final_id'].unique()

    # Create color mapping based on iteration values
    iteration_values = df.groupby('final_id')['iteration'].first()
    min_iter = iteration_values.min()
    max_iter = iteration_values.max()

    # Normalize iteration values to [0, 1] for colorscale
    import plotly.colors as pc
    emrld_colors = pc.sample_colorscale('Emrld',
                                        [(iteration_values[tid] - min_iter) / (max_iter - min_iter)
                                         for tid in unique_ids])
    color_map = {tid: emrld_colors[i] for i, tid in enumerate(unique_ids)}

    # Add a colorbar legend for iterations
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale='Emrld',
            showscale=True,
            cmin=min_iter,
            cmax=max_iter,
            colorbar=dict(
                title="Iteration",
                thickness=15,
                len=0.7
            )
        ),
        hoverinfo='none',
        showlegend=False
    ))

    # Add trajectory lines
    for final_id in unique_ids:
        traj_data = df[df['final_id'] == final_id].sort_values('step')
        t_opacity = 0.2 if (not selected_ids or final_id  not in selected_ids) else 1

        fig.add_trace(go.Scatter(
            x=traj_data['X'],
            y=traj_data['Y'],
            mode='lines',
            line=dict(
                color=color_map[final_id],
                width=2,
            ),
            opacity=t_opacity,
            name=f'Trajectory {final_id}',
            showlegend=False,
            #hoverinfo='skip'
            customdata=traj_data[['image']].values
        ))

    # Add final points
    final_data = df[df['final_object'] == True]
    first = bool(df['istestset'].eq(True).any())
    for final_id in final_data['final_id'].unique():
        fd = final_data[final_data['final_id'] == final_id]
        t_opacity =1 if (not selected_ids or final_id in selected_ids) else 0.2
        fig.add_trace(go.Scatter(
            x=fd['X'],
            y=fd['Y'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=8,
                color=color_map[final_id],
                line=dict(width=1, color='white'),
                opacity=t_opacity,
            ),
            name='Samples',
            showlegend=first,
            #hovertemplate='<b>Final</b><br>SMILES: %{customdata[0]}<extra></extra>',
            customdata=fd[['image', 'final_id']].values
        ))
        first = False

    # Add testset points
    testset_data = df[df['istestset'] == True]
    first = True
    for final_id in testset_data['final_id'].unique():
        fd = testset_data[testset_data['final_id'] == final_id]
        t_opacity = 0.7 if (not selected_ids or final_id in selected_ids) else 0.2
        fig.add_trace(go.Scatter(
            x=fd['X'],
            y=fd['Y'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=5,
                color=px.colors.diverging.curl[9],
                line=dict(width=1, color='white'),
                opacity=t_opacity,
            ),
            name='Testset',
            showlegend=first,
            # hovertemplate='<b>Final</b><br>SMILES: %{customdata[0]}<extra></extra>',
            customdata=fd[['image', 'final_id']].values
        ))
        first = False

    fig.update_traces(
        #customdata=plot_data[['images']].values,
        hoverinfo="none",
        hovertemplate=None,
    )

    # Update layout
    fig.update_layout(
        title="Trajectories in State Space",
        xaxis_title="X",
        yaxis_title="Y",
        hovermode='closest',
        template='plotly_dark',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def update_DAG(
        iteration,
        direction="forward",
        metric="highest",
        max_freq = 0,
        add_handlers = True,
        build_ids=[]
):
    """
    Update DAG visualization based on flow attribute and truncation percentage.

    Parameters:
    -----------
    iteration : list with rang of iterations
    flow_attr : str, Logprobs attribute to use for coloring
    build_ids: list of str, ids of objects of the graph
    Returns: dict : Cytoscape elements and stylesheet
    """

    conn = sqlite3.connect("traindata1/traindata1_1.db")
    placeholders = ",".join("?" for _ in build_ids)
    column = "logprobs_" + direction
    base_where = f"""
        source IN ({placeholders})
        AND target IN ({placeholders})
        AND iteration BETWEEN ? AND ?
    """
    params = build_ids + build_ids + [iteration[0], iteration[1]]

    if metric in ["highest", "lowest"]:
        query = f"""
            WITH edge_ids AS (
                SELECT
                    source,
                    target,
                    GROUP_CONCAT(id, '-') AS id
                FROM edges
                WHERE {base_where}
                GROUP BY source, target
            )
            SELECT
                ei.id,
                e.source,
                e.target,
                {"MAX" if metric == "highest" else "MIN"}(e.{column}) AS metric
            FROM edges e
            JOIN edge_ids ei
              ON ei.source = e.source
             AND ei.target = e.target
            WHERE
                e.source IN ({placeholders})
                AND e.target IN ({placeholders})
                AND e.iteration BETWEEN ? AND ?
            GROUP BY e.source, e.target
        """
        params *= 2

    elif metric == "variance":
        query = f"""
        WITH edge_groups AS (
            SELECT 
                source,
                target,
                MAX(iteration) AS max_iteration,
                AVG({column}) AS mean_metric
            FROM edges
            WHERE source IN ({placeholders})
              AND target IN ({placeholders})
              AND iteration BETWEEN ? AND ?
            GROUP BY source, target
        ),
        edge_ids AS (
            SELECT 
                source,
                target,
                GROUP_CONCAT(id, '-') AS id
            FROM edges
            WHERE source IN ({placeholders})
              AND target IN ({placeholders})
              AND iteration BETWEEN ? AND ?
            GROUP BY source, target
        )
        SELECT 
            ei.id,
            e.source,
            e.target,
            eg.max_iteration AS iteration,
            e.{column} - eg.mean_metric AS metric
        FROM edge_groups eg
        JOIN edges e 
          ON e.source = eg.source
         AND e.target = eg.target
         AND e.iteration = eg.max_iteration
        JOIN edge_ids ei 
          ON ei.source = eg.source
         AND ei.target = eg.target
        WHERE e.source IN ({placeholders})
          AND e.target IN ({placeholders})
          AND e.iteration BETWEEN ? AND ?
        """

        params *= 3

    else:
        query = f"""
            WITH edge_ids AS (
                SELECT
                    source,
                    target,
                    GROUP_CONCAT(id, '-') AS id
                FROM edges
                WHERE {base_where}
                GROUP BY source, target
            )
            SELECT
                ei.id,
                e.source,
                e.target,
                COUNT(*) AS metric
            FROM edges e
            JOIN edge_ids ei
              ON ei.source = e.source
             AND ei.target = e.target
            WHERE
                e.source IN ({placeholders})
                AND e.target IN ({placeholders})
                AND e.iteration BETWEEN ? AND ?
            GROUP BY e.source, e.target
        """
        params *= 2

    edges = pd.read_sql_query(query, conn, params=params)

    query = f"""
        SELECT *
        FROM nodes
        WHERE id IN ({placeholders})
        """
    nodes = pd.read_sql_query(query, conn, params=build_ids)

    # create images as base64
    def encode_image(path):
        if not path:
            return None
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"

    nodes['image'] = nodes['id'].apply(imagefn_from_smiles)

    if add_handlers:
        # get number of children
        query = f"""
                        SELECT
                            source,
                            COUNT(DISTINCT target) AS n_children
                        FROM edges
                        WHERE source IN ({placeholders})
                          AND target NOT IN ({placeholders})
                        GROUP BY source
                    """
        counts = pd.read_sql_query(query, conn, params=build_ids + build_ids)
        nodes = nodes.merge(
            counts,
            left_on='id',
            right_on='source',
            how='left'
        )
        nodes["n_children"] = nodes["n_children"].fillna(0)

        #create handlers
        handler_nodes = nodes[nodes['node_type']!="final"].copy().drop(["image", "reward"], axis=1)
        handler_nodes["node_type"]="handler"
        handler_nodes["id"]="handler_" + handler_nodes["id"]
        handler_nodes["label"] = "Select children: " + handler_nodes["n_children"].astype(int).astype(str)
        handler_nodes["metric"] = metric
        handler_nodes["direction"] = direction
        handler_edges = handler_nodes["id"].copy().to_frame().rename(columns={"id": "target"})
        handler_edges["source"] = handler_edges["target"].str.removeprefix("handler_")

        nodes = pd.concat([nodes, handler_nodes], ignore_index=True)
        edges = pd.concat([edges, handler_edges], ignore_index=True)


    nodes["iteration0"]= iteration[0]
    nodes["iteration1"]= iteration[1]
    if direction == "backward":
        edges.rename(columns={"source": "target", "target": "source"}, inplace=True)


    # convert to cytoscape structure
    nodes = [{"data": row} for row in nodes.to_dict(orient="records")]
    edges = [{"data": row} for row in edges.to_dict(orient="records")]

    conn.close()

    # Compute color scale
    if metric in ['highest', 'lowest']:
        vmin, vmax = -10, 0
        colorscale = px.colors.sequential.Emrld
    elif metric == "variance":
        vmin, vmax = -3, 3
        colorscale = px.colors.diverging.BrBG
    elif metric == "frequency":
        vmin, vmax = 0, max_freq
        colorscale = px.colors.sequential.Emrld
    else:
        vmin, vmax = -1, 1  # fallback
        colorscale = px.colors.sequential.Viridis

    # Create color mapping
    def get_color(value, vmin, vmax, colorscale):
        if np.isnan(value):
            return "#8e8e8e"
        if value < vmin:
            return colorscale[0]  # lowest color
        if value > vmax:
            return colorscale[-1]  # highest color
        if vmax == vmin:
            norm = 0.5
        else:
            norm = (value - vmin) / (vmax - vmin)
        idx = int(norm * (len(colorscale) - 1))
        return colorscale[idx]

    # Build stylesheet
    elements = nodes + edges

    stylesheet = [
        # Default node style
        {
            'selector': 'node',
            'style': {
                'background-color': '#fff',
                'background-image': 'data(image)',
                'background-fit': 'contain',
                'background-clip': 'none',
                'shape': 'round-rectangle',
                'width': '50px',
                'height': '40px',
                'border-width': '1px',
                'border-color': '#000000'
            }
        },
        # START node (keep text label)
        {
            'selector': 'node[node_type = "start"]',
            'style': {
                'background-color': '#BAEB9D',
                'background-image': 'none',  # No image for start node
                'label': f'data(id)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '12px',
                'shape': 'diamond',
                'width': '40px',
                'height': '40px',
                'border-color': '#000000',
                'border-width': '2px',
                'font-weight': 'bold',
                'text-wrap': 'wrap',
                'text-max-width': '55px'
            }
        },
        # Final node
        {
            'selector': 'node[node_type = "final"]',
            'style': {
                'background-color': '#fff',
                'background-image': 'data(image)',
                'background-fit': 'contain',
                'background-clip': 'none',
                'shape': 'round-rectangle',
                'width': '60px',
                'height': '45px',
                'border-width': '3px',
                'border-color': '#000000'
            }
        },
        # Handler node
        {
            'selector': 'node[node_type = "handler"]',
            'style': {
                'background-color': '#fff',
                'background-image': 'none',
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'shape': 'round-rectangle',
                'width': '90px',
                'height': '20px',
                'border-width': '2px',
                'border-color': '#000000',
                #'font-weight': 'bold',
                'text-wrap': 'wrap',
                'text-max-width': '90px'
            }
        },
        # Default edge style
        {
            'selector': 'edge',
            'style': {
                'width': 3,
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 1.5
            }
        }
    ]

    # Add color styles for each edge
    for edge in edges:
        edge_id = edge['data']['id']
        edge_val = edge['data'].get("metric", 0)
        color = get_color(edge_val, vmin, vmax, colorscale)

        stylesheet.append({
            'selector': f'edge[id = "{edge_id}"]',
            'style': {
                'line-color': color,
                'target-arrow-color': color
            }
        })

    return {
        'elements': elements,
        'stylesheet': stylesheet,
    }


def prepare_state_space(data_objects, metadata_to=8):
    """
    Prepares the data for downprojection
    :param data_objects: data_objects
    :param metadata_to: column to seperate metadata from features
    :return: df metadata, df features
    """
    metadata = data_objects.iloc[:, :metadata_to]
    features = data_objects.iloc[:, metadata_to:]
    return metadata.reset_index(drop=True), features.reset_index(drop=True)


def update_state_space(df, selected_ids=[]):
    """
    Updates the state space for final objects
    :param df: dataframe with required columns
    :param selected_ids: list of selected final_ids
    :return: updated plot
    """

    # Separate test set and normal points
    df["total_reward"]=np.sqrt(df["total_reward"])
    df_test = df[df['istestset']]
    df_normal = df[~df['istestset']]

    # Compute opacity
    def compute_opacity(df_sub):
        return [
            0.9 if (not selected_ids or s in selected_ids) else 0.1
            for s in df_sub['final_id']
        ]

    fig = go.Figure()
    sizeref = df["total_reward"].max()*2/(8**2)

    # Normal points with continuous color scale
    fig.add_trace(go.Scatter(
        x=df_normal['X'],
        y=df_normal['Y'],
        mode='markers',
        marker=dict(
            size=df_normal['total_reward'],
            sizeref=sizeref,
            color=df_normal['iteration'],
            colorscale='emrld',
            line=dict(color='black', width=1),
            showscale=True,
            colorbar=dict(
                title="Iteration",
                thickness=15,
                len=0.7
            ),
            opacity=compute_opacity(df_normal),
        ),

        customdata=df_normal[['final_id', 'iteration', 'total_reward', 'image']].values,
        hoverinfo='none',
        name="Samples"
    ))

    # Test set points in red
    if not df_test.empty:
        fig.add_trace(go.Scatter(
            x=df_test['X'],
            y=df_test['Y'],
            mode='markers',
            marker=dict(
                size=df_test['total_reward'],
                sizeref=sizeref,
                color=px.colors.diverging.curl[9],
                line=dict(color='black', width=1),
                opacity=compute_opacity(df_test),
            ),

            customdata=df_test[['final_id', 'iteration', 'total_reward', 'image']].values,
            hoverinfo='none',
            name='Test Set',
        ))

    fig.update_layout(
        autosize=True,
        title=f"Final Objects downprojected<br><sup>Size shows total reward",
        template='plotly_dark',
        legend=dict(
            itemsizing='constant',  # ensures marker size is not scaled
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig


def update_bump(df, n_top, selected_ids, testset_bounds=None):
    """
    Optimized bump chart update for cumulative top-ranked objects
    where rewards are fixed but rank evolves as new objects appear.

    :param df: prepared dataframe (should NOT include testset rows)
    :param n_top: number of top objects to display
    :param selected_ids: list of final_ids to highlight
    :param testset_bounds: optional tuple (min_reward, max_reward) for shading
    :return: Plotly figure
    """

    df_local = df.copy()

    # Remove any testset rows if they exist (safety check)
    if "istestset" in df_local.columns:
        df_local = df_local[~df_local["istestset"]].copy()

    df_local["iteration"] = pd.Categorical(
        df_local["iteration"],
        categories=sorted(df["iteration"].unique()),
        ordered=True
    )

    iterations = df_local["iteration"].cat.categories
    records = []

    # Track objects seen so far and their fixed rewards
    seen_objects = {}

    for it in iterations:
        # Add objects from this iteration
        current_iter_data = df_local[df_local["iteration"] == it]
        for _, row in current_iter_data.iterrows():
            seen_objects[row["text"]] = row["total_reward"]

        # Compute ranks for all seen objects
        # Use text as tiebreaker to ensure stable ordering for equal rewards
        tmp_rank = (
            pd.DataFrame({"text": list(seen_objects.keys()), "total_reward": list(seen_objects.values())})
            .sort_values(["total_reward", "text"], ascending=[False, True])
            .head(n_top)
        )
        tmp_rank["rank"] = range(1, len(tmp_rank) + 1)
        tmp_rank["iteration"] = it

        records.append(tmp_rank[["iteration", "text", "rank", "total_reward"]])

    tmp = pd.concat(records, ignore_index=True)
    tmp = tmp.rename(columns={"rank": "value"})

    # Attach images and IDs
    tmp = tmp.merge(
        df_local[['final_id', 'text', 'image']].drop_duplicates(subset='text'),
        on='text',
        how='left'
    )

    # Precompute marker sizes: circle if object was sampled in that iteration
    tmp["sampled"] = tmp.apply(
        lambda r: 8 if ((df_local["text"] == r["text"]) & (df_local["iteration"] == r["iteration"])).any() else 0,
        axis=1)

    # Sort objects by first appearance rank for consistent line ordering
    first_ranks = tmp.groupby("text")["value"].min().sort_values().index

    fig = go.Figure()

    # Add shading for test set bounds if provided
    if testset_bounds is not None:
        min_reward, max_reward = testset_bounds

        # For each iteration, find the rank bounds
        shade_data = []
        for it in iterations:
            iter_data = tmp[tmp["iteration"] == it].sort_values("value")

            # worst rank (highest number) that still has reward >= max_reward
            at_or_above_max = iter_data[iter_data["total_reward"] >= max_reward]
            rank_above = at_or_above_max["value"].max()-0.5 if not at_or_above_max.empty else 0.5

            # best rank (lowest number) that still has reward <= min_reward
            at_or_below_min = iter_data[iter_data["total_reward"] <= min_reward]
            rank_below = at_or_below_min["value"].min()+0.5 if not at_or_below_min.empty else n_top + 0.5

            shade_data.append({
                "iteration": it,
                "rank_above": rank_above,
                "rank_below": rank_below
            })

        shade_df = pd.DataFrame(shade_data)

        # Add shaded area between the bound ranks
        fig.add_trace(
            go.Scatter(
                x=shade_df["iteration"].tolist() + shade_df["iteration"].tolist()[::-1],
                y=shade_df["rank_above"].tolist() + shade_df["rank_below"].tolist()[::-1],
                fill='toself',
                fillcolor=px.colors.diverging.curl[9],
                opacity=0.5,
                line=dict(width=0),
                showlegend=True,
                hoverinfo='none',
                name='Range of testset rewards'  # Internal name to help identify this trace
            )
        )

    first_iter = (
        tmp.groupby("text")["iteration"]
        .min()
    )
    # Map iteration categories to numeric indices
    iter_to_idx = {it: i for i, it in enumerate(iterations)}
    first_iter_idx = first_iter.map(iter_to_idx)
    emrld = px.colors.sequential.Emrld
    n_colors = len(emrld)
    # Normalize first-iteration index → color
    obj_color = {
        text: emrld[int(idx / max(1, len(iterations) - 1) * (n_colors - 1))]
        for text, idx in first_iter_idx.items()
    }

    for obj in first_ranks:
        obj_df = tmp[tmp["text"] == obj].sort_values("iteration")

        selected_mask = obj_df["final_id"].isin(selected_ids)
        if not selected_ids:
            selected_mask[:] = True
        unselected_mask = ~selected_mask

        for mask, opacity in [(selected_mask, 1), (unselected_mask, 0.1)]:
            sub_df = obj_df[mask]
            if sub_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=sub_df["iteration"],
                    y=sub_df["value"],
                    mode="lines+markers",
                    marker=dict(
                        symbol="circle",
                        size=sub_df["sampled"],
                        color=obj_color[obj],
                    ),
                    line=dict(width=2),
                    opacity=opacity,
                    #name="1",#obj if opacity == 1 else f"{obj} (faded)",
                    customdata=sub_df[['final_id', 'value', 'image', 'total_reward']].values,
                    showlegend=False
                )
            )

    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        autosize=True,
        title=(
            "Replay Buffer Ranked<br><sup>"
            "Sampled Objects ranked by the total reward. "
            "For each Iteration the highest reward objects so far are shown. "
            "Markers show if an object was sampled in this iteration. "
            "</sup>"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        xaxis_title="Iteration",
        yaxis_title="Rank",
        template='plotly_dark',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def update_DAG_overview(direction, metric, iteration):
    column = "logprobs_"+direction
    top_n = 150

    conn = sqlite3.connect("traindata1/traindata1_1.db")

    # Build query based on metric
    if metric == "highest":
        query = f"""
                WITH edge_max AS (
                    SELECT source, target, MAX({column}) as max_val
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    ORDER BY max_val DESC
                    LIMIT {top_n}
                )
                SELECT e.source, e.target, e.iteration, e.{column} as value, em.max_val as metric_val, e.trajectory_id
                FROM edges e
                INNER JOIN edge_max em ON e.source = em.source AND e.target = em.target
                WHERE e.iteration BETWEEN ? AND ?
                ORDER BY em.max_val DESC, e.source, e.target, e.iteration
            """
    elif metric == "lowest":
        query = f"""
                WITH edge_min AS (
                    SELECT source, target, MIN({column}) as min_val
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    ORDER BY min_val ASC
                    LIMIT {top_n}
                )
                SELECT e.source, e.target, e.iteration, e.{column} as value, em.min_val as metric_val, e.trajectory_id
                FROM edges e
                INNER JOIN edge_min em ON e.source = em.source AND e.target = em.target
                WHERE e.iteration BETWEEN ? AND ?
                ORDER BY em.min_val ASC, e.source, e.target, e.iteration
            """
    elif metric == "variance":
        query = f"""
                WITH edge_stats AS (
                    SELECT 
                        source, 
                        target,
                        AVG({column}) as mean_val,
                        COUNT(*) as cnt
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    HAVING cnt > 1
                ),
                edge_variance AS (
                    SELECT 
                        es.source,
                        es.target,
                        es.mean_val,
                        SUM((e.{column} - es.mean_val) * (e.{column} - es.mean_val)) / es.cnt as variance
                    FROM edges e
                    INNER JOIN edge_stats es ON e.source = es.source AND e.target = es.target
                    WHERE e.iteration BETWEEN ? AND ?
                    GROUP BY es.source, es.target, es.mean_val, es.cnt
                    ORDER BY variance DESC
                    LIMIT {top_n}
                )
                SELECT e.source, e.target, e.iteration, e.{column} as value, ev.mean_val, ev.variance as metric_val, e.trajectory_id
                FROM edges e
                INNER JOIN edge_variance ev ON e.source = ev.source AND e.target = ev.target
                WHERE e.iteration BETWEEN ? AND ?
                ORDER BY ev.variance DESC, e.source, e.target, e.iteration
            """
    elif metric == "frequency":
        query = f"""
                WITH edge_freq AS (
                    SELECT source, target, COUNT(*) as freq
                    FROM edges
                    WHERE iteration BETWEEN ? AND ?
                    GROUP BY source, target
                    ORDER BY freq DESC
                    LIMIT {top_n}
                )
                SELECT e.source, e.target, e.iteration, e.{column} as value, ef.freq as metric_val, e.trajectory_id
                FROM edges e
                INNER JOIN edge_freq ef ON e.source = ef.source AND e.target = ef.target
                WHERE e.iteration BETWEEN ? AND ?
                ORDER BY ef.freq DESC, e.source, e.target, e.iteration
            """

    params = iteration*3 if metric=="variance" else iteration*2
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()


    if df.empty:
        return go.Figure().add_annotation(text="No data found", showarrow=False)

    # Create edge identifier and preserve order
    df['edge_id'] = df['source'].astype(str) + '-' + df['target'].astype(str)

    # Get unique edges in order (already ordered by metric in SQL)
    unique_edges = df['edge_id'].unique()
    edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
    df['edge_idx'] = df['edge_id'].map(edge_to_idx)

    # For variance metric, adjust values to be zero-based (value - mean)
    # For frequency metric, use the frequency value itself for coloring
    if metric == "variance":
        edge_means = df.groupby('edge_id')['value'].transform('mean')
        df['plot_value'] = df['value'] - edge_means
    elif metric == "frequency":
        df['plot_value'] = df['metric_val']  # Use frequency for coloring
    else:
        df['plot_value'] = df['value']



    # create edge list and trajectory id list for hover and selection
    trajectory_id_list = df.groupby('edge_idx')['trajectory_id'].agg(list).tolist()
    edge_list = df[['source', "target", "edge_idx"]]
    edge_list = edge_list.drop_duplicates().reset_index(drop=True)
    edge_list = list(edge_list[['source', 'target']].itertuples(index=False, name=None))

    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(
        index='iteration',
        columns='edge_idx',
        values='plot_value',
        aggfunc='first'
    ).sort_index()

    if metric == "variance":
        color_scale = px.colors.diverging.BrBG
        zmin, zmax, zmid = -3, 3, 0
        colorbar_title = "Value - Mean"
        title = f"Edge Heatmap - Highest Variance of {direction.capitalize()} Logprobabilities"
    elif metric == "frequency":
        color_scale = px.colors.sequential.Emrld
        zmin = 0
        zmax = df['metric_val'].max()
        zmid = None
        colorbar_title = "Frequency"
        title = f"Edge Heatmap - Highest frequency"
    else:  # highest or lowest
        color_scale = px.colors.sequential.Emrld
        zmin, zmax, zmid = -10, 0, None
        colorbar_title = "Value"
        title = f"Edge Heatmap - {metric.capitalize()} Value of {direction.capitalize()} Logprobabilities"

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=color_scale,
        showscale=True,
        zmin=zmin,
        zmax=zmax,
        zmid=zmid,
        #customdata=customdata,
        #hovertemplate = "<<%{customdata}>><extra></extra>",
        colorbar=dict(title=dict(text=colorbar_title))
    ))

    fig.update_layout(
        autosize=True,
        template='plotly_dark',
        margin=dict(l=40, r=40, t=40, b=40),
        dragmode="select",
        title=title,
        xaxis=dict(
            title=f"Edges (Top {top_n}, ordered by Metric)",
            showticklabels=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showspikes=False,
        ),
        yaxis=dict(
            title="Iteration",
            showgrid=False,
            showline=False,
            zeroline=False,
            ticks="outside",
            showticklabels=True,
            showspikes=False,
        ),
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    if metric == "frequency":
        return fig, zmax, trajectory_id_list, edge_list
    return fig, None, trajectory_id_list, edge_list

def edge_hover_fig(edge_data):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_data["iteration"],
        y=edge_data["logprobs_forward"],
        mode="lines+markers",
        name="forward"
    ))

    fig.add_trace(go.Scatter(
        x=edge_data["iteration"],
        y=edge_data["logprobs_backward"],
        mode="lines+markers",
        name="backward"
    ))

    fig.update_layout(
        height=200,
        width=250,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(x=0.3, y=1.1, orientation="h", xanchor="center", yanchor="bottom"),
        xaxis_title="Iteration",
        yaxis_title="Logprobs",
        template="plotly_white"
    )

    return fig





# plotting function for molecules.
# keep here for now, when reworking data loading and starting the app pass it when running the app
# make plot utils class and define image_fn on init
from rdkit import Chem
from rdkit.Chem import Draw
import base64


def imagefn_from_smiles(smiles):
    if smiles == "#" or smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmarts(smiles)
        if mol is not None:
            return mol
    except Exception:
        pass

    if mol is None:
       return None

    svg = Draw.MolsToGridImage(
        [mol],
        molsPerRow=1,
        subImgSize=(200, 200),
        useSVG=True
    )
    b64 = base64.b64encode(svg.encode("latin-1")).decode("ascii")

    return f"data:image/svg+xml;base64,{b64}"