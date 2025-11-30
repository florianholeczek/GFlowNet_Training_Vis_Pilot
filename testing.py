import pandas as pd

"""df = pd.read_csv('testset.csv')
print(len(df))
df = df[df["final_object"]==True]
print(len(df))
df = df.drop(["step", "final_object", "iteration", "logprobs_forward", "logprobs_backward"], axis=1)

print(df.head())
df.to_csv('testset.csv', index=False)"""
df = pd.read_csv('train_data.csv')
print(len(df))
df = df[df["step"]!=0]
print(len(df))