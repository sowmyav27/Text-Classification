# Text-Classification

### Agorithm Used: DBSCAN

### Pseudocode:
DBSCAN function: Parameters - Data, density, Eps
For every point in Data --> Calculate its neighbor points
Check the count of neighbors. If its < density — tag the point as a NOISE point
Else append the point to cluster and loop over its neighbors to find Border and CORE
points
if the point in its neighbors > density — tag it as a CORE point, add to cluster. Else tag it as a BORDER point.
Loop over all the points through the above steps


### Normalization technique used:
The data was normalized using sklearn library for l2 normalization.


### Dimensionality reduction:
Truncated SVD was used for dimensionality reduction techniques.
50 components and 100 components were given and tried to get better NMI
