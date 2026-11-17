# Notebooks

* `fact_visualization.ipynb` plots the learned curve against GELU and against
  its own initialisation, the per-epoch coefficient trajectory of the source
  run, and the effect of the harmonic count K.
* `results_analysis.ipynb` loads `results/tables/*.json` and reproduces the
  mean and sample standard deviation tables of the paper together with the
  paired, Holm-corrected comparisons.

Both notebooks execute on CPU and require neither datasets nor
checkpoints; they need `matplotlib` (`pip install -e ".[notebooks]"`).
