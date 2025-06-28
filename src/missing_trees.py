'''
The functionality will handle determining the coordinates of missing trees
within an orchard
'''

'''
1. Need to get the tree data points.
2. Break down trees by id, lat, long, area, and volume.
3. Convert into a 4D array which is k-by-n where n is the number of trees and k is the number of metrics.
4. Investigate the minimum volume, tree entries lenth (n), and if there are 4 that are distinct from the rest.
5. If only existing trees are shown, look for a large gap between a minimum of 3 trees (need a grid or mapping)
6. This gap needs to exceed the average trees area
'''