# LBM

Created to develop a LBM solver prototype for Prof. Paulo Philippi's classes.

### Meetings proposal

Below is the suggested meetings arrangements:

- Meeting 0:
  - [ ] Setup environment (IDEs, extensions, python env)
- Meeting 1:
  - [ ] LBM domain
    - [ ] Nodes creation
    - [ ] Visualization of the nodes
    - [ ] Represent nodes connectivity (indexing scheme)
- Meeting 2:
  - [ ] Model implementation
    - [ ] Velocity set
    - [ ] Equilibrium equation
    - [ ] Collision Operator
- Meeting 3:
  - [ ] Boundary conditions:
    - [ ] Walls
    - [ ] Moving lid
- Meeting 4:
  - [ ] Results:
    - [ ] Export
    - [ ] Visualize
    - [ ] Post processing

### Setup

This project uses a python managed environment for external dependencies.
You can use any dependency manager that you like such as poetry, uv (highly recommended).

```
pip install uv
```

To install the environment:

```
uv pip install .
```

It will then create a .venv folder with the current python environment.
