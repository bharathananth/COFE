
=======================================
Cyclic Ordering with Feature Extraction
=======================================
.. image:: images/coffee_stain.png
   :alt: COFE Logo
   :align: right
   :width: 200px

This package (COFE - *kaa·fee*) implements nonlinear dimensionality reduction with a circular constraint on the (dependent) principal components.

* The manuscript describing and applying COFE is now published in PLoS Biology: https://doi.org/10.1371/journal.pbio.3003196
* Free software: GNU General Public License v3

Features
--------

* Assigns time-labels to high-dimensional data representing an underlying rhythmic process
* Identifies features in the data that contribute to the temporal reordering
* Regularized unsupervised machine learning approach with automated choice of hyperparameters.

Installation
------------

* Prerequisites
   - Python 3.9 or better installed on your system. You can download and install Python from the `official Python website <https://www.python.org/downloads/>`_.
   - Git installed on your system. You can download and install Git from the `official Git website <https://git-scm.com/downloads>`_.
   - Conda installed on your system. You can download and install conda from `official Conda website <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

* Install and 
   #. Open a terminal or command prom 

* Clone the COFE Repository
   #. Open a terminal or command prompt.
   #. Navigate to the directory where you want to install COFE.
   #. Clone the COFE repository from GitHub by running the following command:

   .. code-block:: bash
   
      git clone https://github.com/bharathananth/COFE.git

* Installation
   #. Navigate to the COFE directory:

      .. code-block:: bash
      
         cd COFE

   #. Install and switch to *circular_ordering-env* environment: 

      .. code-block:: bash

        conda env create -f environment.yml
        conda activate circular_ordering-env

   #. You can install COFE and its dependencies by running the following command:

      .. code-block:: bash
   
         python -m pip install .

* Verify Installation
   To verify that COFE is installed correctly, you can try importing it in a Python environment. Open a Python interpreter or create a new Python script, and then try importing COFE:

   .. code-block:: python
   
      import COFE.analyse
      import COFE.plot
      import COFE.scpca

Getting Started
---------------

You can get started with COFE by running it on synthetic data, as illustrated in the Jupyter notebook 
``synthetic_data_example.ipynb`` located in the ``docs/`` folder.

For detailed usage, refer to the docstrings of the COFE functions.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
