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


import sqlite3
df = pd.read_csv('traindata1/traindata1.csv')
print(len(df))
"""conn = sqlite3.connect("traindata1/traindata1.db")
df.to_sql(
    "points",
    conn,
    if_exists="replace",  # or "append"
    index=False
)

cur = conn.cursor()

cur.execute("CREATE INDEX idx_points_finalid ON points(final_id)")
cur.execute("CREATE INDEX idx_points_text ON points(text)")
cur.execute("CREATE INDEX idx_points_iteration ON points(iteration)")
cur.execute("CREATE INDEX idx_points_reward ON points(total_reward)")
cur.execute("CREATE INDEX idx_points_lpforward ON points(logprobs_forward)")
cur.execute("CREATE INDEX idx_points_lpbackward ON points(logprobs_backward)")


conn.commit()
conn.close()"""