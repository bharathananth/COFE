=======================================
Cyclic Ordering with Feature Extraction
=======================================

This package (COFE - *kaa·fee*) implements nonlinear dimensionality reduction with a circular constraint on the (dependent) principal components.

* Preprint: https://doi.org/10.1101/2024.03.13.584582
* Free software: GNU General Public License v3

Features
--------

* Assigns time-labels to high-dimensional data representing an underlying rhythmic process
* Identifies features in the data that contribute to the temporal reordering
* Regularized unsupervised machine learning approach with automated choice of hyperparameters.

Installation
------------

* Prerequisites
   - Python installed on your system. You can download and install Python from the `official Python website <https://www.python.org/downloads/>`_.
   - Git installed on your system. You can download and install Git from the `official Git website <https://git-scm.com/downloads>`_.

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

   #. After installing the dependencies, you can install COFE by running the following command:

      .. code-block:: bash
   
         python setup.py install

* Verify Installation
   To verify that COFE is installed correctly, you can try importing it in a Python environment. Open a Python interpreter or create a new Python script, and then try importing COFE:

   .. code-block:: python
   
      import COFE.analyse
      import COFE.plot
      import COFE.scpca

Usage
-----

Once installed, you can start using COFE in your Python projects. Refer to the COFE documentation or README on the GitHub repository for usage instructions and examples.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
