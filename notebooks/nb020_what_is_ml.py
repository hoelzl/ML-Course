# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# <h1 style="text-align:center;">Machine Learning for Programmers</h1>
# <h2 style="text-align:center;">What is ML?</h2>
# <h3 style="text-align:center;">Dr. Matthias Hölzl</h3>
#

# %% [markdown] slideshow={"slide_type": "slide"}
# # Biological Inspiration
#
# <img src="img/ag/Figure-10-001.png" style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # The Origins of Machine Learning
#
# *A computer can be programmed so that it will learn to play a better game of checkers than can be played by the person who wrote the program.* (Arthur Samuel, 1959)
#
# *Programming computers to learn from experience should eventually eliminate the need for much of this detailed programming effort.* (Arthur Samuel, 1959)

# %% [markdown] slideshow={"slide_type": "subslide"}
# # One Answer (Andrew Glassner)
#
# The phrase *machine learning* describes a growing body of techniques that all have one goal: discover meaningful information from  data. 
#
# Here, “data” refers to anything that can be recorded and measured. [...]
#
# "Meaningful information" is whatever we can extract from the data that  will be useful to us in some way.

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Andrew Glassner's Books
#
# <img src="img/glassner-book.jpg" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-01-001.png">

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-01-002.png">

# %% [markdown] slideshow={"slide_type": "slide"}
# # Another Answer (paraphrasing François Chollet)
#
# - A part of Artificial Intelligence (AI)
# - AI: Making computers solve problems that could previously only be tackled by humans
# - AI doesn't have to involve learning, e.g., expert systems
# - ML: Improving behavior with additional data

# %% [markdown] slideshow={"slide_type": "slide"}
# # Example: MNIST Data
#
# <img src="img/ag/Figure-01-023.png" style="float: right;width: 40%;"/>
#
# We want to recognize hand-written digits:

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Rule-based Systems: Feature Engineering
#
# Extraction of relevant features from data by humans.

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-01-003.png" style="width: 40%; margin-left: auto; margin-right: auto;">

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-01-004.png" style="width: 20%; margin-left: auto; margin-right: auto;">

# %% [markdown] slideshow={"slide_type": "slide"}
# # Supervised Learning (Classification)
#
# (Learning from labeled data)
#
# <img src="img/ag/Figure-01-007.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Training a Classifier
#
# - Show lots of labeled data to a learner
# - Check whether it can correctly classify samples based on features
#
# - Evaluation must be based on different data than training
# - Otherwise the learner could just store the examples it has seen

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Often: Training Loop
#
# <img src="img/ag/Figure-08-001.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Back to MNIST Data
#
# <img src="img/ag/Figure-01-023.png" style="float: right;width: 40%;"/>
#
# Let's try this in practice:

# %% slideshow={"slide_type": "slide"}
