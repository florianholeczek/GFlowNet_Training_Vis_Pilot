import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn import manifold
from umap import UMAP
from collections import defaultdict


def update_state_space_t(data, method, param_value):
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
    metadata_to = 11

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
            customdata=traj_data[['image']].values
        ))

    # Add final points
    final_data = plot_data[plot_data['final_object'] == True]
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
            #hovertemplate='<b>Final</b><br>SMILES: %{customdata[0]}<extra></extra>',
            customdata=fd[['image', 'final_id']].values
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


def collapse_consecutive_texts(df, sum_cols=['logprobs_forward', 'logprobs_backward']):
    """
    Collapse consecutive identical texts within each trajectory.
    Sums specified numeric columns and keeps the last value for other columns.
    """
    collapsed_rows = []

    for final_id, group in df.groupby('final_id'):
        group = group.sort_values('step', ascending=True)

        prev_text = None
        agg_row = None

        for _, row in group.iterrows():
            if row['text'] == prev_text:
                # Aggregate sum columns
                for col in sum_cols:
                    agg_row[col] += row[col]
                # Keep the last value for other columns
                for col in row.index:
                    if col not in sum_cols + ['text', 'final_id']:
                        agg_row[col] = row[col]
            else:
                # Save previous aggregated row
                if agg_row is not None:
                    collapsed_rows.append(agg_row)
                # Start new aggregation
                agg_row = row.to_dict()
                prev_text = row['text']

        # Add the last row
        if agg_row is not None:
            collapsed_rows.append(agg_row)

    return pd.DataFrame(collapsed_rows)


def prepare_graph(df):
    """
    Prepare DAG data structure from trajectory CSV.

    Returns:
    --------
    dict : Dictionary containing nodes and edges data
    """
    print("start prepare")

    # Sort by final_id and step
    df = df.sort_values(['final_id', 'step'], ascending=[True, True])
    df = collapse_consecutive_texts(df)

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
        group = group.sort_values('step', ascending=True)
        prev_node = 'START'

        for idx, row in group.iterrows():
            node_id = row['text']
            step = row['step']
            iteration = row['iteration']
            final_object = row['final_object']

            # Add node if not already added
            if node_id not in node_set:
                nodes.append({
                    'data': {
                        'id': node_id,
                        'label': row['text'],
                        'object': row['text'],
                        'node_type': "final" if final_object else "intermediate",
                        'step': step,
                        'image': f"data:image/svg+xml;base64,{row['image']}",
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
                    'logprobs_forward': row['logprobs_forward'],
                    'logprobs_backward': row['logprobs_backward']
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

        # Find max iteration
        max_iter = max(e['data']['iteration'] for e in edge_list)
        max_iter_edges = [e for e in edge_list if e['data']['iteration'] == max_iter]

        # Latest averages (highest iteration)
        logprobs_forward_latest = sum(
            e['data']['logprobs_forward'] for e in max_iter_edges
        ) / len(max_iter_edges)

        logprobs_backward_latest = sum(
            e['data']['logprobs_backward'] for e in max_iter_edges
        ) / len(max_iter_edges)

        # Overall (all iteration) averages
        logprobs_forward_overall = sum(
            e['data']['logprobs_forward'] for e in edge_list
        ) / len(edge_list)

        logprobs_backward_overall = sum(
            e['data']['logprobs_backward'] for e in edge_list
        ) / len(edge_list)

        # Changes = latest avg - overall avg
        logprobs_forward_change = logprobs_forward_latest - logprobs_forward_overall
        logprobs_backward_change = logprobs_backward_latest - logprobs_backward_overall

        unique_edges.append({
            'data': {
                'id': f"{source}_to_{target}",
                'source': source,
                'target': target,
                'trajectory_id': edge_list[0]['data']['trajectory_id'],
                'logprobs_forward': logprobs_forward_latest,
                'logprobs_backward': logprobs_backward_latest,
                'logprobs_forward_change': logprobs_forward_change,
                'logprobs_backward_change': logprobs_backward_change
            }
        })

    print("done")

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

                total_forward = edge['data'].get('logprobs_forward', 0)
                total_backward = edge['data'].get('logprobs_backward', 0)

                while current not in nodes_to_keep:
                    if len(out_edges[current]) == 0:
                        break

                    next_edge = out_edges[current][0]

                    total_forward += next_edge['data'].get('logprobs_forward', 0)
                    total_backward += next_edge['data'].get('logprobs_backward', 0)

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
                            'truncated': True,
                            'logprobs_forward': total_forward,
                            'logprobs_backward': total_backward
                        }
                    })

    # Filter nodes
    new_nodes = [node for node in nodes if node['data']['id'] in nodes_to_keep]

    return new_nodes, new_edges


def update_DAG(dag_data, flow_attr='logprobs_forward', truncation_pct=0):
    """
    Update DAG visualization based on flow attribute and truncation percentage.

    Parameters:
    -----------
    dag_data : dict
        Dictionary containing 'nodes' and 'edges' from prepare_DAG
    flow_attr : str
        Flow attribute to use for coloring and truncation -> Logprobs
    truncation_pct : float
        Truncation percentage (0-100)

    Returns:
    --------
    dict : Cytoscape elements and stylesheet
    """
    nodes = dag_data['nodes'].copy()
    edges = dag_data['edges'].copy()

    if flow_attr in ['logprobs_backward', 'logprobs_backward_change']:
        for edge in edges:
            edge['data']['source'], edge['data']['target'] = \
                edge['data']['target'], edge['data']['source']

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
    # Fixed ranges by attribute
    if flow_attr in ['logprobs_forward', 'logprobs_backward']:
        vmin, vmax = -8, 0
        colorscale = px.colors.sequential.Emrld
    elif flow_attr in ['logprobs_forward_change', 'logprobs_backward_change']:
        vmin, vmax = -2, 2
        colorscale = px.colors.diverging.BrBG
    else:
        vmin, vmax = -1, 1  # fallback
        colorscale = px.colors.sequential.Viridis

    # Create legend
    legend_fig = create_dag_legend(vmin, vmax, colorscale, flow_attr)

    # Create color mapping
    def get_color(value, vmin, vmax, colorscale):

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
                'label': '',  # Hide label when showing image
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
                'border-width': '3px',
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
        color='iteration',
        size='total_reward',
        opacity=0.9,
        color_continuous_scale="emrld",
        #height=500,
        #width=500,
        #title=f"{method.upper()} Projection"
    )
    fig.update_traces(
        customdata=df[['final_id', 'iteration', 'total_reward', 'image']].values,
        hoverinfo="none",
        hovertemplate=None,
    ),

    fig.update_layout(
        autosize=True,
        title=f"Final Objects downprojected<br><sup>"
              f"Size shows total reward",
        coloraxis_colorbar=dict(title="Iteration")
    )

    return fig




def update_bump(df, n_top):
    """
    Updates the bump chart
    :param df: prepared dataframe
    :param n_top: number of displayed items
    :return: updated plot
    """

    fig = go.Figure()
    df_local = df.copy()
    df_local["iteration"] = pd.Categorical(
        df_local["iteration"],
        categories=sorted(df["iteration"].unique()),
        ordered=True
    )
    iters = df_local["iteration"].cat.categories
    records = []
    seen = set()

    for it in iters:
        this_iter = df_local[df_local["iteration"] <= it]

        # top_n of ALL iterations seen so far
        current_top = (
            this_iter.groupby("text", observed=False)["total_reward"]
            .max()
            .nlargest(n_top)
            .index
        )

        # IMPORTANT: No "seen" anymore
        current_df = this_iter[this_iter["text"].isin(current_top)]

        rank_df = (
            current_df.groupby("text", observed=False)["total_reward"]
            .max()
            .reset_index()
        )

        rank_df["rank"] = rank_df["total_reward"].rank(
            method="first", ascending=False
        )

        rank_df["iteration"] = it

        records.append(rank_df[["iteration", "text", "rank"]])

    tmp = pd.concat(records, ignore_index=True)
    tmp = tmp.rename(columns={"rank": "value"})

    sorted_objects = (
        tmp.groupby("text", observed=False)["value"]
        .min()
        .sort_values()
        .index
    )

    # Attach images for hover/marker logic
    tmp = tmp.merge(
        df_local[['final_id', 'text', 'image', 'total_reward']].drop_duplicates(subset='text'),
        on='text',
        how='left'
    )

    # ----------------- Plot -----------------

    for obj in sorted_objects:

        obj_df = tmp[tmp["text"] == obj].sort_values("iteration")

        # identify where the object is actually present in the iteration
        present_mask = df_local["text"] == obj
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
                        8 if ((df_local["text"] == obj) & (df_local["iteration"] == it)).any()
                        else 0
                        for it in obj_df["iteration"]
                    ],
                    color=px.colors.sequential.Emrld[-1],
                ),
                line=dict(width=2),
                name=obj,
                #marker=dict(symbol='circle-open', size=10),
                customdata=obj_df[['final_id', 'value', 'image', 'total_reward']].values,
                #hovertemplate="Iteration: %{x}<br>Value: %{y}<extra></extra>"
            )
        )

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    ),

    fig.update_layout(
        autosize=True,
        title=f"Replay Buffer Ranked<br><sup>"
              f"Sampled Objects ranked by the total reward. "
              f"For each Iteration it shows the highest reward objects so far. "
              f"Circles show if an object was sampled in this iteration</sup>",
        showlegend=False,
        xaxis_title="Iteration",
        yaxis_title="Rank",
    )

    fig.update_yaxes(autorange="reversed")

    return fig

