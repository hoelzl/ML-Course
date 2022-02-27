# Machine Learning (mostly for Programmers)

Source code and other material for the course "Machine Learning for Python
Programmers".

## Description

This repository contains the source code used during the lectures, as well as templates
and solutions for exercises.

## Installation

In order to set up the necessary environment:

1. Create an Anaconda environment `cam` with the help of [conda] or [mamba]. If you are
   using

   ```bash
   conda env create -f conda-environment.yml
   ```

   Since the conda package resolver is often quite slow you may want to use [mamba], which
   is a C++ implementation of (parts of) `conda`:
      
   ```bash
   mamba env create -f conda-environment.yml
   ```


2. Activate the new environment with:

   ```bash
   conda activate mlcourse
   ```

3. Install the `mlcourse` package by first changing into the `ml-course` dictionary
   (this is the directory that includes the `pyproject.toml` file) and then running `pip
   install`. In order to work with the installed project files, use the `-e` flag:

   ```bash
   cd path/to/ml-course
   pip install -e .
   ```

   (Note that there is a `.` after the `-e` argument.)
   
   

[conda]: https://docs.conda.io/
[mamba]: https://github.com/mamba-org/mamba
