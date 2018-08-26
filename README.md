# Energy-Efficient Smart Buildings by Occupancy Prediction Paper Repo

Based on the increase in energy consumption, energy efficiency has become a paramount topic. Through the energy conservation studies, it can be achieved by predicting occupant presence in buildings. In this study, the accuracy of occupancy prediction has been tested on data from differences in levels of Light, CO2, and Humidity in intervals of 1, 3, and 5 minutes. Artificial Neural Networks (ANN), Support Vector Machines (SVM), and Decision Trees have been chosen as learning algorithms.


## Requirements

* tensorflow
* numpy
* pandas
* sklearn
* scipy
* matplotlib

Please use the command below to extract png version of decision trees.

```sh
dot -Tpng tree.dot -o tree.png
```


