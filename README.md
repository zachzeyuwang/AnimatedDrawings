# AnimatedDrawings

This is a python reimplementation of [Animated Construction of Line Drawings](http://sweb.cityu.edu.hk/hongbofu/projects/animatedConstructionOfLineDrawings_SiggA11/) by Fu et al. published on ACM Transaction on Graphics (Proceedings of SIGGRAPH Asia 2011).

1. `compute_costs.py`: reads drawings from `drawing.json` and computes unary and binary stroke costs, output saved in `cost.json`.
2. `construct_graphs.py`: reads computed `cost.json` and constructs a graph for each drawing, output saved in `graph.json`.
3. `tsp_bnb.py`: reads `graph.json` and searches the Hamiltonian path (similar to the traveling salesman problem) using branch and bound, output saved in `result.json` with stroke ordering that minimizes the cost.

Note: `drawing.json` is formatted as a dictionary. You can access a drawing with `drawing_json[image_key][artist_key]`. A drawing is a list of strokes. A stroke is a dictionary with keys of path, pressure, color, width, and opacity. path is a string with all timestamps, x and y coordinates concatenated. The canvas is 800x800.
