import numpy as np

NX = 8
h = 1.0 / NX

nodes_x, nodes_y = [], []
node_set = set()

for i in range(NX + 1):
    for j in range(NX + 1):
        x, y = i * h, j * h
        if x > 0.5 and y < 0.5: continue
        if x > 0.75 and y > 0.75: continue
        node_set.add((i, j))
        nodes_x.append(x)
        nodes_y.append(y)

nodes_x = np.array(nodes_x)
nodes_y = np.array(nodes_y)
node_map = {(i, j): idx for idx, (i, j) in enumerate(node_set)}

# Find which cell contains (0.25, 0.25)
# (0.25, 0.25) = (2*h, 2*h) so i=2, j=2
i, j = 2, 2
xc, yc = (i + 0.5) * h, (j + 0.5) * h
print(f"Cell center at ({i},{j}): ({xc}, {yc})")
print(f"Is in subdomain B? xc>0.5 and yc<0.5: {xc > 0.5 and yc < 0.5}")
print(f"Is in notch? xc>0.75 and yc>0.75: {xc > 0.75 and yc > 0.75}")

# The cell (2,2) has corners (2,2), (3,2), (2,3), (3,3)
# In grid coordinates: (0.25, 0.25), (0.375, 0.25), (0.25, 0.375), (0.375, 0.375)
# All these should be in subdomain A

corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
print(f"Corners: {corners}")
for c in corners:
    if c in node_map:
        print(f"  {c} -> node {node_map[c]} at ({nodes_x[node_map[c]]}, {nodes_y[node_map[c]]})")
    else:
        print(f"  {c} NOT in node_map!")

# Check all cells around (0.25, 0.25)
print("\nChecking cells around (0.25, 0.25):")
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        ii, jj = i + di, j + dj
        if 0 <= ii < NX and 0 <= jj < NX:
            xc, yc = (ii + 0.5) * h, (jj + 0.5) * h
            in_B = xc > 0.5 and yc < 0.5
            in_notch = xc > 0.75 and yc > 0.75
            status = "SKIP(B)" if in_B else ("SKIP(notch)" if in_notch else "KEEP")
            print(f"  Cell ({ii},{jj}): center=({xc:.3f},{yc:.3f}) -> {status}")
