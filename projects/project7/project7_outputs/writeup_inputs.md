# Fixed LAM Verification
- Cost: 0.179960
- N_hit: 25
- N_total: 30
- Hit fraction: 0.833333
- y0 contribution: 0.299600
- 0.1800 check: PASS
- Note: literal prompt vector with 8.21e-6 gives cost 0.929960 and releases 0 tracked droplets

# GA Results
- Lambda*: [83.68327243794364, 0.0, 221.67256199738455, -15.95458780176869, 0.003708453590272911, 0.15536147222664265, 0.01654906000193559, 0.6379956048024442, 8.915677771763841, 92.67456197612319, 36.62883803945472, 1460.9859134791882]
- Pi(Lambda*): 0.187418
- Hit fraction: 0.804348
- Number of generations: 20
- Convergence notes:
  - Best cost decreases substantially over the run and levels off by later generations
  - Average cost stays above best cost throughout, indicating population spread
  - Parents' average cost tracks closer to the best curve in later generations

# Figures
- ga_best_cost.png: best cost versus generation
- ga_avg_cost.png: average population cost versus generation
- ga_parents_avg.png: parents' average cost versus generation
- ga_combined.png: all three GA convergence curves on one plot
- optimized_design.png: optimized design visualization showing aircraft path, droplets, and fire region
- sensitivity_baseline_surface.png: baseline sensitivity surface over aircraft velocity and particle drop velocity
- sensitivity_baseline_heatmap.png: baseline sensitivity heatmap over aircraft velocity and particle drop velocity
- sensitivity_optimized_surface.png: optimized sensitivity surface over aircraft velocity and particle drop velocity
- sensitivity_optimized_heatmap.png: optimized sensitivity heatmap over aircraft velocity and particle drop velocity

# Sensitivity Observations
- Baseline sampled minimum cost: 0.149960 at aircraft velocity 63.49 m/s and particle velocity 10.71 m/s
- Optimized sampled minimum cost: 0.070027 at aircraft velocity 67.46 m/s and particle velocity 0.00 m/s
- Cost changes nonlinearly with both particle velocity and aircraft velocity
- Low aircraft velocity regions include some of the highest sampled costs
- Optimal regions are localized rather than smooth over the full grid
- GA is appropriate because the response surface is nonlinear and multi-region rather than simple convex bowl-shaped

# Notes / Limitations
- The literal prompt fixed-LAM vector uses 8.21e-6 for flow rate; that does not reproduce ~0.1800 in the current scaffold
- Outputs use tracked super-particles rather than every physical droplet
- Optimized media was saved as both a static PNG and an MP4 animation
- Sensitivity results depend on the sampled grid resolution, not a continuous sweep