import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn import manifold
from umap import UMAP
from collections import defaultdict


def update_state_space_t(data_trajectories, method, param_value, n_trajectories):
    """
    Create a trajectory visualization in state space with dimensionality reduction.

    Parameters:
    -----------
    data_trajectories : pd.DataFrame
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
    metadata_to = 11

    # Filter trajectories
    data = data_trajectories[
        data_trajectories["final_id"] <= min(n_trajectories, max(data_trajectories["final_id"]))].copy()

    # Split metadata and features
    metadata = data.iloc[:, :metadata_to]
    features = data.iloc[:, metadata_to:]

    # Apply dimensionality reduction
    if method == 'tsne':
        reduced = manifold.TSNE(perplexity=param_value, random_state=42).fit_transform(features)
    elif method == "umap":
        reducer = UMAP(n_neighbors=param_value, random_state=42)
        reduced = reducer.fit_transform(features)
    else:
        raise NotImplementedError('Method not implemented')

    # Combine back with metadata
    plot_data = pd.concat([
        metadata.reset_index(drop=True),
        pd.DataFrame(reduced, columns=['X', 'Y'])
    ], axis=1)

    # Create figure
    fig = go.Figure()

    """# Get unique trajectory IDs and assign colors
    unique_ids = plot_data['final_id'].unique()
    colors = px.colors.qualitative.Dark24
    color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(unique_ids)}

    # Add trajectory lines
    for final_id in unique_ids:
        traj_data = plot_data[plot_data['final_id'] == final_id].sort_values('step')

        fig.add_trace(go.Scatter(
            x=traj_data['X'],
            y=traj_data['Y'],
            mode='lines',
            line=dict(color=color_map[final_id], width=2),
            opacity=0.15,
            name=f'Trajectory {final_id}',
            showlegend=False,
            hoverinfo='skip'
        ))"""
    # Get unique trajectory IDs and their iterations
    unique_ids = plot_data['final_id'].unique()

    # Create color mapping based on iteration values
    iteration_values = plot_data.groupby('final_id')['iteration'].first()
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
        traj_data = plot_data[plot_data['final_id'] == final_id].sort_values('step')

        fig.add_trace(go.Scatter(
            x=traj_data['X'],
            y=traj_data['Y'],
            mode='lines',
            line=dict(color=color_map[final_id], width=2),
            opacity=0.15,
            name=f'Trajectory {final_id}',
            showlegend=False,
            #hoverinfo='skip'
            customdata=traj_data[['images']].values
        ))

    # Add start points
    start_data = plot_data[plot_data['state'] == 'start']
    for final_id in start_data['final_id'].unique():
        fd = start_data[start_data['final_id'] == final_id]
        fig.add_trace(go.Scatter(
            x=fd['X'],
            y=fd['Y'],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color=color_map[final_id],
                line=dict(width=1, color='white')
            ),
            name=f'Start {final_id}',
            showlegend=False,
            hovertemplate='<b>Start</b><br>SMILES: %{customdata[0]}<extra></extra>',
            customdata=fd[['images']].values
        ))

    # Add final points
    final_data = plot_data[plot_data['state'] == 'final']
    for final_id in final_data['final_id'].unique():
        fd = final_data[final_data['final_id'] == final_id]
        fig.add_trace(go.Scatter(
            x=fd['X'],
            y=fd['Y'],
            mode='markers',
            marker=dict(
                symbol='square',
                size=8,
                color=color_map[final_id],
                line=dict(width=1, color='white')
            ),
            name=f'Final {final_id}',
            showlegend=False,
            hovertemplate='<b>Final</b><br>SMILES: %{customdata[0]}<extra></extra>',
            customdata=fd[['images']].values
        ))

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
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def create_dag_legend(vmin, vmax, colorscale, flow_attr):
    """
    Create a plotly figure showing a colorbar legend for the DAG.

    Parameters:
    -----------
    vmin : float
        Minimum value for the colorscale
    vmax : float
        Maximum value for the colorscale
    colorscale : list
        Plotly colorscale
    flow_attr : str
        Name of the flow attribute

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Create a dummy scatter plot with just the colorbar
    fig = go.Figure(data=go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            showscale=True,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title=dict(
                    text=flow_attr.replace('_', ' ').title(),
                    side='right'
                ),
                thickness=20,
                len=0.9,
                x=0.5,
                xanchor='center'
            )
        ),
        hoverinfo='none'
    ))

    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def prepare_DAG(csv_path, n_trajectories=8):
    """
    Prepare DAG data structure from trajectory CSV.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing trajectory data
    n_trajectories : int
        Number of trajectories to include

    Returns:
    --------
    dict : Dictionary containing nodes and edges data
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Limit trajectories
    unique_ids = df['final_id'].unique()[:n_trajectories]
    df = df[df['final_id'].isin(unique_ids)]

    # Sort by final_id and step (descending to go from high to low step numbers)
    df = df.sort_values(['final_id', 'step'], ascending=[True, False])

    # Create nodes and edges
    nodes = []
    edges = []
    node_set = set()

    # Add START node
    nodes.append({
        'data': {
            'id': 'START',
            'label': '#',
            'node_type': 'start',
            'step': -1
        }
    })
    node_set.add('START')

    # Process each trajectory
    for final_id, group in df.groupby('final_id'):
        group = group.sort_values('step', ascending=False)
        prev_node = 'START'

        for idx, row in group.iterrows():
            node_id = row['smiles']
            step = row['step']
            state = row['state']
            iteration = row['iteration']

            # Add node if not already added
            if node_id not in node_set:
                node_label = row['smiles']
                if len(node_label) > 20:
                    node_label = node_label[:17] + '...'

                nodes.append({
                    'data': {
                        'id': node_id,
                        'label': node_label,
                        'smiles': row['smiles'],
                        'node_type': "final" if state == "final" else "intermediate",
                        'step': step,
                        'valid': row['valid'],
                        'image': f"data:image/svg+xml;base64,{row['images']}",
                    }
                })
                node_set.add(node_id)

            # Add edge from previous node
            edge_id = f"{prev_node}_to_{node_id}"
            edges.append({
                'data': {
                    'id': edge_id,
                    'source': prev_node,
                    'target': node_id,
                    'trajectory_id': final_id,
                    'iteration': iteration,
                    'flow_forward': row['flows_forward'],
                    'flow_backward': row['flows_backward']
                }
            })

            prev_node = node_id

    # Deduplicate edges and compute flow statistics
    edge_groups = defaultdict(list)
    for edge in edges:
        key = (edge['data']['source'], edge['data']['target'])
        edge_groups[key].append(edge)

    unique_edges = []
    for (source, target), edge_list in edge_groups.items():
        max_iter = max(e['data']['iteration'] for e in edge_list)
        min_iter = min(e['data']['iteration'] for e in edge_list)

        max_iter_edges = [e for e in edge_list if e['data']['iteration'] == max_iter]
        flow_forward = sum(e['data']['flow_forward'] for e in max_iter_edges) / len(max_iter_edges)
        flow_backward = sum(e['data']['flow_backward'] for e in max_iter_edges) / len(max_iter_edges)

        min_iter_edges = [e for e in edge_list if e['data']['iteration'] == min_iter]
        flow_forward_min = sum(e['data']['flow_forward'] for e in min_iter_edges) / len(min_iter_edges)
        flow_backward_min = sum(e['data']['flow_backward'] for e in min_iter_edges) / len(min_iter_edges)

        unique_edges.append({
            'data': {
                'id': f"{source}_to_{target}",
                'source': source,
                'target': target,
                'trajectory_id': edge_list[0]['data']['trajectory_id'],
                'flow_forward': flow_forward,
                'flow_backward': flow_backward,
                'flow_forward_change': flow_forward - flow_forward_min,
                'flow_backward_change': flow_backward - flow_backward_min
            }
        })

    return {'nodes': nodes, 'edges': unique_edges}


def truncate_linear_chains(nodes, edges, edges_to_keep_ids):
    """
    Remove intermediate nodes that only have one edge in and one edge out.
    Only applies to edges not in edges_to_keep_ids.

    Parameters:
    -----------
    nodes : list
        List of node dictionaries
    edges : list
        List of edge dictionaries
    edges_to_keep_ids : set
        Set of edge IDs that should not be truncated

    Returns:
    --------
    tuple : (truncated_nodes, truncated_edges)
    """
    # Build adjacency information
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    node_types = {}

    for edge in edges:
        source = edge['data']['source']
        target = edge['data']['target']
        out_edges[source].append(edge)
        in_edges[target].append(edge)

    for node in nodes:
        node_types[node['data']['id']] = node['data']['node_type']

    # Identify nodes to keep
    nodes_to_keep = set()
    for node in nodes:
        node_id = node['data']['id']
        node_type = node['data']['node_type']

        # Always keep start and final nodes
        if node_type in ['start', 'final']:
            nodes_to_keep.add(node_id)
        # Keep nodes with multiple incoming or outgoing edges
        elif len(in_edges[node_id]) != 1 or len(out_edges[node_id]) != 1:
            nodes_to_keep.add(node_id)
        # Keep nodes connected to edges we must keep
        else:
            for edge in in_edges[node_id] + out_edges[node_id]:
                if edge['data']['id'] in edges_to_keep_ids:
                    nodes_to_keep.add(node_id)
                    break

    # Build new edge list
    new_edges = []
    processed_chains = set()

    for node_id in nodes_to_keep:
        for edge in out_edges[node_id]:
            target = edge['data']['target']
            edge_id = edge['data']['id']

            # If edge must be kept or target is kept, add as-is
            if edge_id in edges_to_keep_ids or target in nodes_to_keep:
                new_edges.append(edge)
            else:
                # Follow the chain
                current = target
                chain_start = node_id
                trajectory_id = edge['data']['trajectory_id']

                chain_id = f"{chain_start}_{current}"
                if chain_id in processed_chains:
                    continue
                processed_chains.add(chain_id)

                while current not in nodes_to_keep:
                    if len(out_edges[current]) == 0:
                        break
                    next_edge = out_edges[current][0]
                    current = next_edge['data']['target']

                # Create truncated edge
                if current != target:
                    new_edge_id = f"{chain_start}_to_{current}_truncated"
                    new_edges.append({
                        'data': {
                            'id': new_edge_id,
                            'source': chain_start,
                            'target': current,
                            'trajectory_id': trajectory_id,
                            'truncated': True
                        }
                    })

    # Filter nodes
    new_nodes = [node for node in nodes if node['data']['id'] in nodes_to_keep]

    return new_nodes, new_edges


def update_DAG(dag_data, flow_attr='flow_forward', truncation_pct=0):
    """
    Update DAG visualization based on flow attribute and truncation percentage.

    Parameters:
    -----------
    dag_data : dict
        Dictionary containing 'nodes' and 'edges' from prepare_DAG
    flow_attr : str
        Flow attribute to use for coloring and truncation
    truncation_pct : float
        Truncation percentage (0-100)

    Returns:
    --------
    dict : Cytoscape elements and stylesheet
    """
    nodes = dag_data['nodes'].copy()
    edges = dag_data['edges'].copy()

    # Calculate flow values and determine edges to keep
    flow_values = []
    for edge in edges:
        flow_val = edge['data'].get(flow_attr, 0)
        flow_values.append(abs(flow_val))

    edges_to_keep = set()
    if truncation_pct < 100:
        # Keep top (100 - truncation_pct)% of edges
        keep_pct = (100 - truncation_pct) / 100
        threshold_idx = int(len(flow_values) * keep_pct)

        if threshold_idx > 0:
            sorted_flows = sorted(flow_values, reverse=True)
            threshold = sorted_flows[min(threshold_idx - 1, len(sorted_flows) - 1)]

            for edge in edges:
                if abs(edge['data'].get(flow_attr, 0)) >= threshold:
                    edges_to_keep.add(edge['data']['id'])

    # Apply truncation if needed
    if truncation_pct > 0:
        nodes, edges = truncate_linear_chains(nodes, edges, edges_to_keep)

    # Compute color scale for non-truncated edges
    non_truncated_flows = [
        edge['data'].get(flow_attr, 0)
        for edge in edges
        if not edge['data'].get('truncated', False)
    ]

    if non_truncated_flows:
        vmin = min(non_truncated_flows)
        vmax = max(non_truncated_flows)
    else:
        vmin, vmax = 0, 1

    # Select colorscale based on flow attribute
    if flow_attr in ['flow_forward', 'flow_backward']:
        colorscale = px.colors.sequential.Emrld
    else:  # flow_forward_change or flow_backward_change
        colorscale = px.colors.diverging.BrBG

    # Create legend
    legend_fig = create_dag_legend(vmin, vmax, colorscale, flow_attr)

    # Create color mapping
    def get_color(value, vmin, vmax, colorscale):
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
                'label': '',  # Hide label when showing image
                'shape': 'round-rectangle',
                'width': '40px',
                'height': '40px',
                'border-width': '1px',
                'border-color': '#616161'
            }
        },
        # START node (keep text label)
        {
            'selector': 'node[node_type = "start"]',
            'style': {
                'background-color': '#BAEB9D',
                'background-image': 'none',  # No image for start node
                'label': 'data(label)',
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
        # Final node (show image)
        {
            'selector': 'node[node_type = "final"]',
            'style': {
                'background-color': '#fff',
                'background-image': 'data(image)',
                'background-fit': 'contain',
                'background-clip': 'none',
                'label': '',  # Hide label for final nodes
                'shape': 'round-rectangle',
                'width': '60px',
                'height': '45px',
                'border-width': '2px',
                'border-color': '#000000'
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
        if edge['data'].get('truncated', False):
            color = '#999999'
        else:
            flow_val = edge['data'].get(flow_attr, 0)
            color = get_color(flow_val, vmin, vmax, colorscale)

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
        'legend': legend_fig
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


def update_state_space(metadata, features, method="umap", param_value=15):
    """
    Updates the state space for final objects
    :param metadata: df metadata
    :param features: df features
    :param method: downprojection method
    :param param_value: n_neighbors for umap, perplexity for tsne
    :return: updated plot
    """

    # Downprojection
    if method == "tsne":
        proj = manifold.TSNE(
            perplexity=param_value,
            init='pca',
            learning_rate='auto'
        ).fit_transform(features)
    elif method == "umap":
        reducer = UMAP(n_neighbors=param_value)
        proj = reducer.fit_transform(features)
    else:
        raise NotImplementedError("Method not implemented")

    df = pd.concat([metadata, pd.DataFrame(proj, columns=['X','Y'])], axis=1)

    # Create Plot
    fig = px.scatter(
        df,
        x='X',
        y='Y',
        color='reward_total',
        #size='reward_total',
        #hover_data=['smiles', 'reward_total'],
        opacity=0.9,
        color_continuous_scale="emrld",
        #height=500,
        #width=500,
        #title=f"{method.upper()} Projection"
    )
    fig.update_traces(
        customdata=df[['iteration', 'reward_total', 'images']].values,
        hoverinfo="none",
        hovertemplate=None,
    ),

    fig.update_layout(
        autosize=True,
        coloraxis_colorbar=dict(title="Reward")
    )

    return fig


def prepare_bump(data):
    """
    Prepares the data for the bump chart.
    takes a dataframe and returns the dataframe and the color map
    """
    df = data.copy()
    df["iteration"] = pd.Categorical(
        df["iteration"],
        categories=sorted(df["iteration"].unique()),
        ordered=True
    )

    # Create color map
    palette = px.colors.qualitative.Dark24
    all_objects = df["smiles"].unique()
    color_map = {obj: palette[i % len(palette)] for i, obj in enumerate(sorted(all_objects))}

    return df, color_map


def update_bump(df, n_top):
    """
    Updates the bump chart
    :param df: prepared dataframe
    :param n_top: number of displayed items
    :return: updated plot
    """

    fig = go.Figure()
    df_local = df.copy()
    iters = df_local["iteration"].cat.categories
    records = []
    seen = set()

    for it in iters:
        this_iter = df_local[df_local["iteration"] <= it]

        # top_n of ALL iterations seen so far
        current_top = (
            this_iter.groupby("smiles", observed=False)["reward_total"]
            .max()
            .nlargest(n_top)
            .index
        )

        current_top = (
            this_iter.groupby("smiles", observed=False)["reward_total"]
            .max()
            .nlargest(n_top)
            .index
        )

        # IMPORTANT: No "seen" anymore
        current_df = this_iter[this_iter["smiles"].isin(current_top)]

        rank_df = (
            current_df.groupby("smiles", observed=False)["reward_total"]
            .max()
            .reset_index()
        )

        rank_df["rank"] = rank_df["reward_total"].rank(
            method="first", ascending=False
        )

        rank_df["iteration"] = it

        records.append(rank_df[["iteration", "smiles", "rank"]])

    tmp = pd.concat(records, ignore_index=True)
    tmp = tmp.rename(columns={"rank": "value"})

    sorted_objects = (
        tmp.groupby("smiles", observed=False)["value"]
        .min()
        .sort_values()
        .index
    )

    # Attach images for hover/marker logic
    tmp = tmp.merge(
        df_local[['smiles', 'images', 'reward_total']].drop_duplicates(subset='smiles'),
        on='smiles',
        how='left'
    )

    # ----------------- Plot -----------------

    for obj in sorted_objects:

        obj_df = tmp[tmp["smiles"] == obj].sort_values("iteration")

        # identify where the object is actually present in the iteration
        present_mask = df_local["smiles"] == obj
        present_iters = set(df_local[present_mask]["iteration"])

        markers = [
            "circle" if it in present_iters else None
            for it in obj_df["iteration"]
        ]

        fig.add_trace(
            go.Scatter(
                x=obj_df["iteration"],
                y=obj_df["value"],
                mode="lines+markers",
                marker=dict(
                    symbol="circle",
                    size=[
                        10 if ((df_local["smiles"] == obj) & (df_local["iteration"] == it)).any()
                        else 0
                        for it in obj_df["iteration"]
                    ]
                ),
                name=obj,
                #marker=dict(symbol='circle-open', size=10),
                customdata=obj_df[['value', 'images', 'reward_total']].values,
                #hovertemplate="Iteration: %{x}<br>Value: %{y}<extra></extra>"
            )
        )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    ),

    fig.update_layout(
        autosize=True,
        title=f"Replay buffer ranked",
        showlegend=False,
        xaxis_title="Iteration",
        yaxis_title="Rank",
    )

    fig.update_yaxes(autorange="reversed")

    return fig

