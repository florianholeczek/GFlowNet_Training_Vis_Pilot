import base64
import sqlite3

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class Plotter:

    def __init__(self, data, image_fn):
        self.data = data
        self.image_fn = image_fn

    def update_DAG(
            self,
            iteration,
            direction="forward",
            metric="highest",
            max_freq=0,
            add_handlers=True,
            build_ids=[]
    ):
        """
        Updates the DAG based on the given metric and direction.
        :param iteration: iteration range
        :param direction: forward/backward
        :param metric: lowest, highest, variance, frequency
        :param max_freq: Highest frequency in the data
        :param add_handlers: add handlers if in expanding mode, dont add in selection mode
        :param build_ids: states to build the dag from
        :return: dict with elements and stylesheets
        """

        conn = sqlite3.connect(self.data)
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

        nodes['image'] = nodes['id'].apply(self.image_fn)

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

            # create handlers
            handler_nodes = nodes[nodes['node_type'] != "final"].copy().drop(["reward"], axis=1)
            handler_nodes["node_type"] = "handler"
            handler_nodes["id"] = "handler_" + handler_nodes["id"]
            handler_nodes["label"] = "Select children: " + handler_nodes["n_children"].astype(int).astype(str)
            handler_nodes["metric"] = metric
            handler_nodes["direction"] = direction
            handler_edges = handler_nodes["id"].copy().to_frame().rename(columns={"id": "target"})
            handler_edges["source"] = handler_edges["target"].str.removeprefix("handler_")

            nodes = pd.concat([nodes, handler_nodes], ignore_index=True)
            edges = pd.concat([edges, handler_edges], ignore_index=True)

        nodes["iteration0"] = iteration[0]
        nodes["iteration1"] = iteration[1]
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
                    # 'font-weight': 'bold',
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

    def update_DAG_overview(self, direction, metric, iteration):
        column = "logprobs_" + direction
        top_n = 150

        conn = sqlite3.connect(self.data)

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

        params = iteration * 3 if metric == "variance" else iteration * 2
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
            # customdata=customdata,
            # hovertemplate = "<<%{customdata}>><extra></extra>",
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

    def edge_hover_fig(self, edge_data):
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

    def update_state_space(self, df, selected_ids=[], metric="total_reward"):
        """
        Updates the state space for final objects
        :param df: dataframe with required columns text, x, y, metric, istetset, iteration
        :param selected_ids: list of selected texts
        :return: updated plot
        """

        # Normalize metric, scale to range 6-30px, set size=4 for missing values (no metric in testset)
        m_min = df[metric].min()
        m_max = df[metric].max()
        df["metric_norm"] = 6 + (df[metric] - m_min) / (m_max - m_min) * (30 - 6)
        df["metric_norm"] = df["metric_norm"].fillna(4)

        # Separate test set and normal points
        df_test = df[df['istestset']]
        df_normal = df[~df['istestset']]

        # Compute opacity
        def compute_opacity(df_sub):
            return [
                0.9 if (not selected_ids or s in selected_ids) else 0.1
                for s in df_sub['text']
            ]

        fig = go.Figure()

        # Normal points with continuous color scale
        fig.add_trace(go.Scatter(
            x=df_normal['x'],
            y=df_normal['y'],
            mode='markers',
            marker=dict(
                size=df_normal["metric_norm"],
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

            customdata=df_normal[['iteration', metric, 'text']].values,
            hoverinfo='none',
            name="Samples"
        ))

        # Test set points in red
        if not df_test.empty:
            fig.add_trace(go.Scatter(
                x=df_test['x'],
                y=df_test['y'],
                mode='markers',
                marker=dict(
                    size=df_test["metric_norm"],
                    color=px.colors.diverging.curl[9],
                    line=dict(color='black', width=1),
                    opacity=compute_opacity(df_test),
                ),

                customdata=df_test[['iteration', metric, 'text']].values,
                hoverinfo='none',
                name='Test Set',
            ))

        fig.update_layout(
            autosize=True,
            title=f"Final Objects downprojected<br><sup>Size shows {metric} for the latest iteration the object occured",
            template='plotly_dark',
            legend=dict(
                itemsizing='constant',  # ensures marker size is not scaled
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )

        return fig

    def update_bump_all(self, df, n_top, selected_ids, order):
        """
        Optimized bump chart update for cumulative top-ranked objects
        where rewards are fixed but rank evolves as new objects appear.

        :param df: prepared dataframe (should NOT include testset rows)
        :param n_top: number of top objects to display
        :param selected_ids: list of final_ids to highlight
        :param order: ASC or DSC for highest/lowest rank
        :return: Plotly figure
        """

        df["iteration"] = pd.Categorical(
            df["iteration"],
            categories=sorted(df["iteration"].unique()),
            ordered=True
        )

        iterations = df["iteration"].cat.categories
        records = []

        # Track objects seen so far and their fixed rewards
        seen_objects = {}

        for it in iterations:
            # Add objects from this iteration
            current_iter_data = df[df["iteration"] == it]
            for _, row in current_iter_data.iterrows():
                seen_objects[row["text"]] = row["metric"]

            # Compute ranks for all seen objects
            # Use text as tiebreaker to ensure stable ordering for equal rewards
            asc = [False, True] if order == "DESC" else [True, True]
            tmp_rank = (
                pd.DataFrame({"text": list(seen_objects.keys()), "metric": list(seen_objects.values())})
                .sort_values(["metric", "text"], ascending=asc)
                .head(n_top)
            )
            tmp_rank["rank"] = range(1, len(tmp_rank) + 1)
            tmp_rank["iteration"] = it

            records.append(tmp_rank[["iteration", "text", "rank", "metric"]])

        tmp = pd.concat(records, ignore_index=True)
        tmp = tmp.rename(columns={"rank": "value"})

        # Attach IDs
        tmp = tmp.merge(
            df[['final_id', 'text']].drop_duplicates(subset='text'),
            on='text',
            how='left'
        )

        # Precompute marker sizes: circle if object was sampled in that iteration
        tmp["sampled"] = tmp.apply(
            lambda r: 8 if ((df["text"] == r["text"]) & (df["iteration"] == r["iteration"])).any() else 0,
            axis=1)

        # Sort objects by first appearance rank for consistent line ordering
        first_ranks = tmp.groupby("text")["value"].min().sort_values().index

        fig = go.Figure()

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
                        line=dict(width=1),
                        opacity=opacity,
                        # name="1",#obj if opacity == 1 else f"{obj} (faded)",
                        customdata=sub_df[['final_id', 'value', 'metric', 'text']].values,
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

    def update_bump_iter(self, df, n_top, selected_ids, order):
        """
        Optimized bump chart update for cumulative top-ranked objects
        where rewards are fixed but rank evolves as new objects appear.

        :param df: prepared dataframe (should NOT include testset rows)
        :param n_top: number of top objects to display
        :param selected_ids: list of final_ids to highlight
        :param order: ASC or DSC for highest/lowest rank
        :return: Plotly figure
        """

        df = df.drop_duplicates(subset=["iteration", "text", "metric"])
        df = df.rename(columns={"rank": "oldrank"})
        df['rank'] = df.groupby('iteration')['oldrank'] \
            .rank(method='dense', ascending=True).astype(int)

        # Create Scatter plot
        fig = go.Figure(
            go.Scatter(
                x=df['iteration'],
                y=df['rank'],
                mode='markers',
                marker=dict(color=df['iteration'], colorscale='Emrld', size=10),
                line=dict(width=1),
                customdata=df[['final_id', 'rank', 'metric', 'text']].values,
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