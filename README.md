# equity-pairs-trading

Quantitave Project - Developing & Simulating Equities Pairs Trading Strategies

Currently finished components:
- Complete backtesting system in <code>src/</code>
- Robust data acquisition and handling module: <code>src/data_handler.py</code>
- Pair research pipeline: <code>notebooks/pair_research.ipynb</code>

Working on:
- Signal generation models (each model will be developed and tested in <code>notebooks/</code>)

Planned signal generation models:  
- Model 1: Static Ornstein-Uhlenbeck (OU) with Monte Carlo-optimized thresholds  
-	Model 2: Dynamic OU estimation with stochastic volatility; dynamic thresholds via stopping theory  
- Model 3: Dynamic OU plus ML volatility-regime classification for adaptive thresholding  

Project write up and repo break down (including reproducability guide) coming soon...
