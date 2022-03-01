#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Rekurrente Netze (RNNs)
#

# %% [markdown]
# ## Sequentialle Daten
#
# <img src="img/ag/Figure-22-001.png" style="width: 10%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Floating Window
#
# <img src="img/ag/Figure-22-002.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Verarbeitung mit MLP
#
# <img src="img/ag/Figure-22-002.png" style="width: 20%; margin-left: 10%; margin-right: auto; float: left;"/>
# <img src="img/ag/Figure-22-003.png" style="width: 35%; margin-left: 10%; margin-right: auto; float: right;"/>

# %% [markdown]
# ## MLP berücksichtigt die Reihenfolge nicht!
#
# <img src="img/ag/Figure-22-004.png" style="width: 25%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## RNNs: Netzwerke mit Speicher
#
# <img src="img/ag/Figure-22-005.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-006.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-007.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-008.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Zustand: Reperatur-Roboter
#
# <img src="img/ag/Figure-22-009.png" style="width: 85%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Arbeitsweise RNN
#
# <img src="img/ag/Figure-22-010.png" style="width: 85%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## State wird nach der Verarbeitung geschrieben
#
# <img src="img/ag/Figure-22-011.png" style="width: 35%; margin-left: 10%; margin-right: auto; float: left;"/>
# <img src="img/ag/Figure-22-012.png" style="width: 15%; margin-left: auto; margin-right: 10%; float: right;"/>

# %% [markdown]
# ## Netzwerkstruktur (einzelner Wert)
#
# Welche Operation ist sinnvoll?
#
# <img src="img/ag/Figure-22-013.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Netzwerkstruktur (einzelner Wert)
#
# <img src="img/ag/Figure-22-014.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Repräsentation in Diagrammen
#
# <img src="img/ag/Figure-22-015.png" style="width: 10%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Entfaltete Darstellung
#
# <img src="img/ag/Figure-22-016.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Netzwerkstruktur für mehrere Werte
#
# <img src="img/ag/Figure-22-018.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Darstellung der Daten
#
# <img src="img/ag/Figure-22-019.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# #  Darstellung der Daten
#
# <img src="img/ag/Figure-22-020.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# #  Darstellung der Daten
#
# <img src="img/ag/Figure-22-021.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Arbeitsweise
#
# <img src="img/ag/Figure-22-022.png" style="width: 55%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Probleme
# <div style="margin-top: 20pt; float:left;">
#     <ul>
#         <li>Verlust der Gradienten</li>
#         <li>Explosion der Gradienten</li>
#         <li>Vergessen</li>
#     </ul>
# </div>
#
# <img src="img/ag/Figure-22-023.png" style="width: 55%; margin-left: auto; margin-right: 5%; float: right;"/>

# %% [markdown]
#
# ## LSTM
#
# <img src="img/ag/Figure-22-029.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>


# %% [markdown]
# ## Gates
#
# <img src="img/ag/Figure-22-024.png" style="width: 55%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Gates
#
# <img src="img/ag/Figure-22-025.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Forget-Gate
#
# <img src="img/ag/Figure-22-026.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Remember Gate
#
# <img src="img/ag/Figure-22-027.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Output Gate
#
# <img src="img/ag/Figure-22-028.png" style="width: 55%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## LSTM
#
# <img src="img/ag/Figure-22-029.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-030.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-031.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-032.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## LSTM Funktionsweise
#
# <img src="img/ag/Figure-22-033.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Verwendung von LSTMs
#
# <img src="img/ag/Figure-22-034.png" style="width: 75%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Darstellung von LSTM Layern
#
# <img src="img/ag/Figure-22-035.png" style="width: 25%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Conv/LSTM (Conv/RNN) Architektur
#
# <img src="img/ag/Figure-22-036.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Tiefe RNN Netze
#
# <img src="img/ag/Figure-22-037.png" style="width: 55%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Bidirektionale RNNs
#
# <img src="img/ag/Figure-22-038.png" style="width: 65%; margin-left: auto; margin-right: auto;"/>


# %% [markdown]
# ## Tiefe Bidirektionale Netze
#
# <img src="img/ag/Figure-22-039.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Anwendung: Generierung von Text
#
# <img src="img/ag/Figure-22-040.png" style="width: 15%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Trainieren mittels Sliding Window
#
# <img src="img/ag/Figure-22-042.png" style="width: 25%; margin-left: auto; margin-right: auto;"/>


# %% [markdown]
# # Vortrainierte LSTM-Modelle

# %%
from fastai.text.all import *

# %%
path = untar_data(URLs.IMDB)
path.ls()

# %%
(path/'train').ls()

# %%
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')

# %%
dls.show_batch()

# %%
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
# learn.fine_tune(4, 1e-2)

# %%
learn.fine_tune(4, 1e-2)

# %%
learn.show_results()

# %%
learn.predict("I really liked that movie!")

# %% [markdown]
# # ULMFiT
#
# Problem: Wir trainieren die oberen Layer des Classifiers auf unser Problem, aber das Language-Model bleibt auf Wikipedia spezialisiert!
#
# Lösung: Fine-Tuning des Language-Models bevor wir den Classifier trainieren.
#
# <img src="img/ulmfit.png" style="width: 75%; margin-left: auto; margin-right: auto;"/>

# %%
dls_lm = TextDataLoaders.from_folder(path, is_lm=True, valid_pct=0.1)

# %%
dls_lm.show_batch(max_n=5)

# %%
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], path=path, wd=0.1).to_fp16()

# %%
learn.fit_one_cycle(1, 1e-2)

# %%
learn.save('epoch-1')

# %%
learn = learn.load('epoch-1')

# %%
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3)

# %%
learn.save_encoder('finetuned')

# %%
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]

# %%
print("\n".join(preds))

# %%
dls_clas = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', text_vocab=dls_lm.vocab)

# %%
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
learn = learn.load_encoder('finetuned')

# %%
learn.fit_one_cycle(1, 2e-2)

# %%
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))

# %%
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

# %%
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))

# %%
