import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms.traversal import dfs_edges
from sklearn import manifold
from umap import UMAP
from collections import defaultdict


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
        template='plotly',
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


def prepare_graph(df, flow_attr, truncation_pct):
    """
    Prepare DAG data structure from trajectory CSV.

    Returns:
    --------
    dict : Dictionary containing nodes and edges data
    """

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
            'step': -1,
            'image': None,
            'reward': None,
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
                        'node_type': "final" if final_object else "intermediate",
                        'step': step,
                        'image': f"data:image/svg+xml;base64,{row['image']}",
                        'reward': row['total_reward'],
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
                'edge_type': 'standard',
                'logprobs_forward': logprobs_forward_latest,
                'logprobs_backward': logprobs_backward_latest,
                'logprobs_forward_change': logprobs_forward_change,
                'logprobs_backward_change': logprobs_backward_change
            }
        })

    # from prepare

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
        nodes, edges = truncate_linear_chains(nodes, unique_edges, edges_to_keep)


    return {'nodes': nodes, 'edges': edges}



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
                total_forward_change = edge['data'].get('logprobs_forward_change', 0)
                total_backward_change = edge['data'].get('logprobs_backward_change', 0)

                while current not in nodes_to_keep:
                    if len(out_edges[current]) == 0:
                        break

                    next_edge = out_edges[current][0]

                    total_forward += next_edge['data'].get('logprobs_forward', 0)
                    total_backward += next_edge['data'].get('logprobs_backward', 0)
                    total_forward_change += next_edge['data'].get('logprobs_forward_change', 0)
                    total_backward_change += next_edge['data'].get('logprobs_backward_change', 0)

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
                            'edge_type': 'truncated',
                            'logprobs_forward': total_forward,
                            'logprobs_backward': total_backward,
                            'logprobs_forward_change': total_forward_change,
                            'logprobs_backward_change': total_backward_change
                        }
                    })

    # Filter nodes
    new_nodes = [node for node in nodes if node['data']['id'] in nodes_to_keep]

    return new_nodes, new_edges


def update_DAG(dag_data, flow_attr='logprobs_forward', built_ids=[]):
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

    built_nodes = [{
        'data': {
            'id': 'START',
            'label': '#',
            'node_type': 'start',
            'step': -1,
        }
    }]
    built_edges = []
    built_node_ids = ['START',]
    child_counter = defaultdict(list)
    for node in nodes:
        if node['data']['id'] in built_ids:
            built_nodes.append(node)
            built_node_ids.append(node['data']['id'])
    for edge in edges:
        if edge['data']['source'] in built_node_ids:
            if edge['data']['target'] in built_node_ids:
                built_edges.append(edge)
            else:
                child_counter[edge['data']['source']].append({
                    'id': edge['data']['target'],
                    'metric': edge['data'][flow_attr]
                })

    for k,v in child_counter.items():
        child_data = []
        for child in v:
            print(child)
            """for node in nodes:
                if child['id'] == node['data']['id']:
                    print(node['data'])
                    child_data.append({
                        'id': child['id'],
                        'metric': child['metric'],
                        'final': node['data']['node_type']=='final',
                        'image': node['data']['image'],
                        'reward': node['data']['reward'],
                    })
                    break"""
        built_nodes.append({
                    'data': {
                        'id': k+"selector",
                        'node_type': "handler",
                        'label': f"Other: {len(v)} children ▾",
                        'children': v,
                    }
                })
        built_edges.append({
                        'data': {
                            'id': f"{k}_handler",
                            'source': k,
                            'target': k+"selector",
                            'edge_type': 'handler',
                        }
                    })






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
    edges = built_edges
    nodes = built_nodes
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
                'label': f'data(label)',
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
        flow_val = edge['data'].get(flow_attr, 0)
        if edge['data']['edge_type'] == 'handler':
            color = '#000000'
        else:
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
        template='plotly',
        legend=dict(
            itemsizing='constant',  # ensures marker size is not scaled
        )
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

            # Find the line just BELOW the max_reward threshold
            # We want the worst rank (highest number) that still has reward >= max_reward
            at_or_above_max = iter_data[iter_data["total_reward"] >= max_reward]
            rank_above = at_or_above_max["value"].max()-0.5 if not at_or_above_max.empty else 0.5

            # Find the line just ABOVE the min_reward threshold
            # We want the best rank (lowest number) that still has reward <= min_reward
            at_or_below_min = iter_data[iter_data["total_reward"] <= min_reward]
            rank_below = at_or_below_min["value"].min()+0.5 if not at_or_below_min.empty else n_top + 0.5

            shade_data.append({
                "iteration": it,
                "rank_above": rank_above,
                "rank_below": rank_below
            })

        shade_df = pd.DataFrame(shade_data)

        # Add shaded area between the bound ranks
        # Use a separate trace name to help Dash identify it
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

    # Plot the ranked objects
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
                        color=px.colors.sequential.Emrld[-1],
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
        template='plotly'
    )

    fig.update_yaxes(autorange="reversed")

    return fig