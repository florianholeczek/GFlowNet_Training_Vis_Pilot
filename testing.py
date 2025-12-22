import pandas as pd
import base64
import os
from cairosvg import svg2png
from tqdm import tqdm
import plotly.express as px

"""df = pd.read_csv('testset.csv')
print(len(df))
df = df[df["final_object"]==True]
print(len(df))
df = df.drop(["step", "final_object", "iteration", "logprobs_forward", "logprobs_backward"], axis=1)

print(df.head())
df.to_csv('testset.csv', index=False)"""
"""df = pd.read_csv('train_data2.csv')
df2 = df[df["final_id"]==90]
df2["iteration"] = 99
print(df2)
dfn = pd.concat([df, df2])
dfn.to_csv("train_data2.csv", index=False)"""
#print(len(px.colors.diverging.curl))
"""def save_svg_as_png(base64_svg, filename):
    output_dir = "traindata1/images"
    # Remove data URI prefix if present
    if base64_svg.startswith("data:image"):
        base64_svg = base64_svg.split(",", 1)[1]

    svg_bytes = base64.b64decode(base64_svg)

    png_path = os.path.join(output_dir, filename)
    svg2png(bytestring=svg_bytes, write_to=png_path, output_width=200, output_height=200)

    return png_path


import sqlite3
df = pd.read_csv('traindata1/train_data.csv')
for idx, row in tqdm(df.iterrows()):
    png_filename = f"image_{idx}.png"
    #png_path = save_svg_as_png(row["image"], png_filename)
    output_dir = "traindata1/images"
    png_path = os.path.join(output_dir, png_filename)
    df.at[idx, "image"] = png_path

df.to_csv("traindata1.csv", index=False)"""

# remove duplicate texts and save as db
"""import sqlite3
df = pd.read_csv('traindata1/traindata1.csv')
cols = list(df.columns)

# Identify the logprob columns and their positions
logprob_cols = ["logprobs_forward", "logprobs_backward"]
logprob_positions = [cols.index(c) for c in logprob_cols]


# 1. Ensure last = highest step
df = df.sort_values(["final_id", "text", "step"])

# 2. Sum logprobs per (trajectory_id, text)
logprob_sums = (
    df.groupby(["final_id", "text"], as_index=False)[
        ["logprobs_forward", "logprobs_backward"]
    ]
    .sum()
)

# 3. Take the last row per (trajectory_id, text) for ALL other columns
last_rows = (
    df.groupby(["final_id", "text"], as_index=False)
      .last()
      .drop(columns=["logprobs_forward", "logprobs_backward"])
)

# 4. Merge summed logprobs back
result = last_rows.merge(
    logprob_sums,
    on=["final_id", "text"],
    how="left"
)

# --- restore column order ---
# Remove logprob columns temporarily
for c in logprob_cols:
    result.pop(c)

# Insert them back at their original positions
for c, pos in sorted(zip(logprob_cols, logprob_positions), key=lambda x: x[1]):
    result.insert(pos, c, logprob_sums[c].values)


# 5. Optional: restore trajectory ordering
result = result.sort_values(["final_id", "step"]).reset_index(drop=True)
df = result
df.to_csv("traindata1_1.csv", index=False)
print(len(df))
conn = sqlite3.connect("traindata1/traindata1_1.db")
df.to_sql(
    "trajectories",
    conn,
    if_exists="replace",  # or "append"
    index=False
)

cur = conn.cursor()

cur.execute("CREATE INDEX idx_points_finalid ON trajectories(final_id)")
cur.execute("CREATE INDEX idx_points_text ON trajectories(text)")
cur.execute("CREATE INDEX idx_points_iteration ON trajectories(iteration)")
cur.execute("CREATE INDEX idx_points_reward ON trajectories(total_reward)")
cur.execute("CREATE INDEX idx_points_lpforward ON trajectories(logprobs_forward)")
cur.execute("CREATE INDEX idx_points_lpbackward ON trajectories(logprobs_backward)")


conn.commit()
conn.close()"""



"""
import dash
from dash import html
import dash_cytoscape as cyto
import sqlite3
import pandas as pd

# --- Load data from SQLite ---
conn = sqlite3.connect("traindata1/traindata1_1.db")
nodes_df = pd.read_sql("SELECT * FROM nodes", conn)
edges_df = pd.read_sql("SELECT * FROM edges", conn)
conn.close()

# --- Prepare Cytoscape elements ---
elements = []

# Nodes
for _, row in nodes_df.iterrows():
    node_data = {
        'data': {'id': row['id'], 'label': row['id']},
        'position': {'x': 0, 'y': 0}  # position will be auto-layout
    }
    # Use image if exists, otherwise white node
    if row['image'] and row['image'].strip():
        node_data['style'] = {
            'background-image': row['image'],
            'background-fit': 'cover',
            'border-width': 2,
            'border-color': '#000'
        }
    else:
        node_data['style'] = {
            'background-color': '#ffffff',
            'border-width': 2,
            'border-color': '#000'
        }
    elements.append(node_data)

# Edges
for _, row in edges_df.iterrows():
    elements.append({
        'data': {'source': row['source'], 'target': row['target']},
        'classes': 'directed'
    })

# --- Dash app ---
app = dash.Dash(__name__)
cyto.load_extra_layouts()

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        layout={'name': 'breadthfirst'},  # DAG-specific layout
        style={'width': '100%', 'height': '800px'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': '',  # hide labels
                    'width': '50px',
                    'height': '50px'
                }
            },
            {
                'selector': 'edge.directed',
                'style': {
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 2,
                    'line-color': '#888',
                    'target-arrow-color': '#888'
                }
            }
        ]
    )
])

if __name__ == '__main__':
    app.run(debug=True)

"""


import pandas as pd

# Example data
df = pd.DataFrame({
    'trajectory_id': [10, 5, 20, 20, 20, 30],
    'id': [1, 1, 2, 2, 2, 3],
    'logprobs_forward': [-0.2, -0.4, -0.1, -0.3, -0.2, -0.5],
    'logprobs_backward': [-0.1, -0.2, -0.3, -0.4, -0.2, -0.6]
})

# Group by trajectory_id and id, then take mean of logprobs
df_agg = df.groupby(['trajectory_id', 'id'], as_index=False)[['logprobs_forward', 'logprobs_backward']].mean()

print(df_agg)