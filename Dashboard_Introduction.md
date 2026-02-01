# Introduction to the dashboard

## Sidebar

The sidebar has three sections:
  - View lets you toggle between the visualizations for the final objects (state space and ranking) and for the DAG
    
  - General has controls independet of the view.
  
    - In almost all visualizations you can select states to see them highlighted in the other ones. "Clear selection" resets all these selections.
   
    - Iterations lets you filter the final states and trajectories to include only these sampled inside the range.
      All Visualizations will be recomputed to this range. A full range is useful to see the progress, set the range only to the last iterations to see the samples of the current state of model training.
    
  - Final Objects / DAG has controls depending on the view

Final Objects:
  - Object metric: Choose between reward, loss and additionally logged metrics. Both the State Space Plot and the Rank Plot will use the selected metric

  - Ranking Metric: Objects in the Rank Plot will be ranked based on this. Highest over all shows for each iteration the highest ranking final Objects from this iteration and all before it.This allows to see changes in the highest ranking ones over time. Highest per Iter shows the highest ranking final objects of the specific iteration, ignoring previous iterations. lowest over all / lowest per iter reverses the ranking order.

  - Use Testset: If True and a testset was provided in the data this allows a comparison between samples and testset objects in the State Space Plot

  - State Space Style: Set how to display the state space:
    
    - Hex Ratio shows the downprojected samples in hexagon bins. If no testset is used coloring is based on the number of samples in each bin. I a testset is used coloring shows if the ratio of the number of samples and testset objects deviates from the global ratio. If areas of the state space contain testset objects but no or few samples this means that the model struggles to sample in that part of the state space (assuming enough samples are provided)
   
    - Hex Metric shows coloring based on the average value of the selected Object Metric of the samples of each bin
   
    - Scatter shows all samples without aggregation
   
  - Dimensinality reduction: Choose t-sne or umap to downproject the data

  - Choose the perplexity (tsne) or n_neighbors (umap) for the downprojection

DAG:
  - Layout: Graph Layout

  - Direction: Choose to look at the forward or backward Logprobabilities

  - Metric: How to color the edges (In the DAG and the Overview). Highest and Lowest colors the edges based on the logprobabilites and influences the ordering direction of the overview. Variance computes for all edges the difference: Logprobability - Mean Logprobability of the same edge. This shows if the Logprob of an edge increases / decreases during training. Frequency shows how often an edge appears in the samples.

## Visualizations

