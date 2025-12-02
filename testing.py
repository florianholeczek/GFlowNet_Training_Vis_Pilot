import pandas as pd
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
print(len(px.colors.diverging.curl))
