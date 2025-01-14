import nbformat

with open('path/to/notebook.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

with open('path/to/fixed_notebook.ipynb', 'w') as f:
    nbformat.write(nb, f)
