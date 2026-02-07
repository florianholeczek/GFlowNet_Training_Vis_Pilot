# Introduction to the dashboard

## Sidebar

The sidebar has three sections:
  - 'View' lets you toggle between the visualizations for the final objects (state space and ranking) and for the DAG
    
  - 'General' has controls independet of the view.
  
    - In all visualizations you can select states to see them highlighted in the other ones. 'Clear selection' resets all these selections.
   
    - 'Iterations' lets you filter the final states and trajectories to include only these sampled inside the range.
      All Visualizations will be recomputed to this range. A full range is useful to see the progress, set the range only to the last iterations to see the samples of the current state of model training.
    
  - 'Final Objects' / 'DAG' has controls depending on the view

Final Objects:
  - Object metric: Choose between reward, loss and additionally logged metrics. Both the State Space Plot and the Rank Plot will use the selected metric

  - Ranking Metric: Objects in the Rank Plot will be ranked based on this. 'Highest over all' shows for each iteration the highest ranking final Objects from this iteration and all before it.This allows to see changes in the highest ranking ones over time. Highest per Iter shows the highest ranking final objects of the specific iteration, ignoring previous iterations. lowest over all / lowest per iter reverses the ranking order.

  - Use Testset: If True and a testset was provided in the data this allows a comparison between samples and testset objects in the State Space Plot

  - State Space Style: Set how to display the state space:
    
    - 'Hex Ratio' shows the downprojected samples in hexagon bins. If no testset is used coloring is based on the number of samples in each bin. I a testset is used coloring shows if the ratio of the number of samples and testset objects deviates from the global ratio. If areas of the state space contain testset objects but no or few samples this means that the model struggles to sample in that part of the state space (assuming enough samples are provided)
   
    - 'Hex Metric' shows coloring based on the average value of the selected Object Metric of the samples of each bin
   
    - 'Hex Corr.' shows the correlation between log reward and forward logprobabilities. 

    - 'Scatter' shows all samples without aggregation. Size shows the total reward, coloring shows the last iteration this object was sampled
   
  - Dimensinality reduction: Choose 't-sne' or 'umap' to downproject the data

  - Choose the perplexity (tsne) or n_neighbors (umap) for the downprojection

DAG:
  - Layout: Graph Layout

  - Direction: Choose to look at the forward or backward logprobabilities

  - Metric: How to color the edges (In the DAG and the Overview). 'Highest' and 'Lowest' colors the edges based on the logprobabilites and influences the ordering direction of the overview. 'Variance' computes for all edges the difference: logprobability - mean logprobability of the same edge. This shows if the logprobability of an edge increases / decreases during training. 'Frequency' shows how often an edge appears in the samples.

## Visualizations

### Ranking

<img width="2578" height="966" alt="grafik" src="https://github.com/user-attachments/assets/619cb87a-3fc1-439a-bcb7-0da6cfe81804" />

This plot shows the Top 30 Objects based on the choosen metric (reward, loss or custom). Markers show that an object was sampled in this iteration. Lines show the changes in the objects rank over iterations (Only if 'Highest /Lowest over all' is choosen as Ranking Metric). 

You can see here when the model discovered high reward areas. You can see what Objects have the highest Loss (Here the model struggles to sample proportional to the reward). The objects with the lowest loss show precise modelling by the policy.

Hovering over and objects shows its metric and the state image. You can select Objects via the Plotly Lasso Tool in the upper right. This selects not only via markers but also lines.

### State space

<img width="2525" height="2812" alt="Bildschirmfoto vom 2026-02-07 13-30-093" src="https://github.com/user-attachments/assets/42d0932a-a5ac-4861-9911-9c9e3e4227ed" />

Based on the provided / computed features a downprojection was applied to show the state space in two dimensions. Based on the setting in 'State Space Style', all objects are shown in a scatterplot or the space is binned in hexbins and coloring happens based on an aggreagation metric.

Coloring by 'State Space Style':

  - Hex Ratio without testset: If no testset is used, the plot simply shows the frequency of samples in each bin. Note that the dimensionality reduction does not neccesarily preserve density, so without the testset this plot might be misleading in the differentiation from dense / sparse bins: A bin with many samples is not neccessarily dense in the original state space. Use the hover information to see if the states of a bin are really similar.

  - Hex Ratio with testset: If a testset is provided, the plot shows coloring based on the number of sampled objects and testset objects in each bin. The Odds Ratio is computed: (n_samples/n_testset) / (n_samples_total/n_testset_total). This is to account for the different number of samples and testset objects. The Odds Ratio is then scaled to [-1,1] via tanh(log(OR)). So a bin with -1 consists of only testset objects and a bin with 1 of only samples. Bins with a metric of -1 might be the most interesting ones. This part of the state space is represented in the testset, but the model fails to sample from it. Use the information on hover to see how the states look like (see example below).

  - Hex Obj. Metric: This shows the bins based on the average of the metric choosen in 'Object Metric' (reward, loss or custom logged metric) of the samples (testset is ignored). Areas with high loss might be interesting, as here the model fails to sample proportionally to reward.

  - Hex Correlation: This shows the correlation between the sum of the forward logprobabilities and the log reward of each bin of the samples (testset is ignored). The higher the correlation the better the model is in sampling proportional to reward in this area of the state space. Bins with less than 10 samples are greyed out as the correlation would not be very robust. You can still see the scatterplot for them on hover.

  - Scatter: Objects are colored based on the last iteration they were sampled. The size is based on the choosen Object Metric. 

Click on a hexbin to select it and see its samples in detail and highlighted in the other visualizations. If the State Space Style is 'Scatter' you can use the Lasso Tool to select objects.

Hovering over an object point shows its state image, the iteration it was sampled last and the choosen Object Metric.
Hovering over an hexbin  shows:

  1. The mean loss of all its samples over iterations and the range of all losses.

  2. A Histogram over the reward of the samples and the testset (if used). The y-Axis is density (area = 1), to allow comparison of the distribution of samples and testset objects.

  3. If a custom logged metric is choosen in Object Metric and 'Hex Obj. Metric' is choosen, a histogram similar to that of the reward is displayed

  4. If 'Hex Correlation' is choosen as State Space Style, a scatterplot is shown plotting the log reward vs the sum of the forward logprobabilities of each sample. A second plot shows the reward vs the product of the forward probabilities.

  5. The result of the state aggregation function provided when running the dashboard. This shows what all the states have in common (Most common substructure, areas present in all states in a grid environment, point groups shared in a crytal env..., its up to you how to implement it in your env). This allows checking a bin to what part of the state space it represents.

  6. The number of samples in a bin, the number of testset objects in a bin and the value of the metric used for coloring.

Example:

In the case of the image above, all the upper purple hexbins have the same most common substructure (The three rings). These bins almost exlusively have testset objects. This shows that the model might fail to sample objects containing this structure or it might underrepresent this part of the state space (assuming the testset is representative).

### DAG - Edge Overview

<img width="866" height="2758" alt="Bildschirmfoto vom 2026-02-07 13-43-17" src="https://github.com/user-attachments/assets/53397f8a-497d-4187-8c39-2b4b9fd33c0b" />

This heatmap shows the Top 150 Ranked edges based on the choosen metric (highest logprobabilities, lowest logprobabilities, variance, frequency) for the choosen direction. On the y-axis are the ranked edges, you can see its source and target state on hover. If an edge is sampled in an iteration it gets colored based on the choosen metric, otherwise it is black. Use the buttons above and below the plot to see the edges not covered in the Top 150. This visualization is only filtered by the iterations slider and not by other selections. 

You can see which transitions are most probable (highest logprobabilities) and for which transitions the probabilities change the most (variance) during training - the value shows for each transition difference of the logprobability of the transition in this iteration to the mean logprobability of this transition over all (selected) iterations.

Use the box select to see all trajectories with this transition in the DAG and all final objects of these trajectories in the other visualizations. Hover over an edge to see its source and target state, the value of the choosen metric and how its logprobabilities changed during training. 

### DAG

<img width="4690" height="3001" alt="Bildschirmfoto vom 2026-02-07 17-22-46" src="https://github.com/user-attachments/assets/7dd0c80b-ad13-4ceb-95ec-37a736059d81" />

This shows a subgraph of the Directed Acyclic Graph of all trajectories for the current selection. There are two methods of selecting:

  1. By selections in other plots: If you select final objects or edges in other plots, all trajectories with these final objects or edges will be shown here.

  2. By exploring: each node has a handler (Select children: N). Clicking it shows its children in the table on the right. selecting them adds them to the subgraph, deselection removes them and all their children if they are not connected to other nodes. Selecting a node by clicking on it changes to the first method: All trajectories with this node are shown. Clicking 'Clear selection' gets you back to exploring again.

Selecting the root node collapses the built graph and clears all selections. The edge coloring is based on the choosen metric and uses the same colorscale as the Edge Heatmap to the left. Transitions with one child and one parent have been merged to avoid long chains, their logprobabilities have been added up. For each transition, only one edge is displayed, the color is based on the latest iteration.


This visualization allows to explore how the final states are built and how the forward and backward policies differ.
