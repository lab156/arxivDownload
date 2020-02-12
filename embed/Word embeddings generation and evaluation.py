# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np

# %load_ext autoreload
# %autoreload 2
from report_lengths import generate
# -

sample_dict = {str(p): -p for p in range(5)}

ordered = generate(sample_dict)

for o in ordered:
    print(o[])

sample_dict.values()
