# Compiling PARROT's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/). Sphinx imports PARROT from this repository to document its code, so it needs PARROT's dependencies as well as Sphinx and the ReadTheDocs theme. `requirements.txt` lists them (it is what Read the Docs installs), so from this directory run

```bash
pip install sphinx -r requirements.txt
```

Once installed, you can use the `Makefile` in this directory to compile static HTML pages by

```bash
make html
```

The compiled docs will be in the `_build/html` directory and can be viewed by opening `_build/html/index.html`.
