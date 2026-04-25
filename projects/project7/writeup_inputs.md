# Project 7 Writeup Inputs

- Random seed used: 144
- Fixed LAM requested in scaffold gives cost 0.9300 and releases 0 tracked particles.
- Assignment-consistent fixed LAM benchmark gives cost 0.1800, hit fraction 0.8333, released particles 30, aircraft height term 0.2996.
- Final best Lambda*: [ 8.368327e+01,  0.000000e+00,  2.216726e+02, -1.595459e+01,  3.708454e-03,
  1.553615e-01,  1.654906e-02,  6.379956e-01,  8.915678e+00,  9.267456e+01,
  3.662884e+01,  1.460986e+03]
- Final best cost Pi(Lambda*): 0.187418
- Best-design hit fraction: 0.804348
- Best-design total particles released: 46

- Figures to include:
  - ga_best_cost_vs_generation.png: best GA cost versus generation.
  - ga_average_cost_vs_generation.png: population average cost versus generation.
  - ga_parents_average_cost_vs_generation.png: surviving parents' average cost versus generation.
  - ga_convergence_combined.png: all three GA convergence curves together.
  - optimized_design_screenshot.png: representative optimized-design simulation frame.
  - default_sensitivity_surface.png and default_sensitivity_heatmap.png: sensitivity sweep using the default baseline design.
  - optimized_sensitivity_surface.png and optimized_sensitivity_heatmap.png: sensitivity sweep using the GA-best design.

- Plot observations:
  - Default sensitivity minimum cost in sampled grid: 0.1500 at plane velocity 63.49 m/s and drop velocity 10.71 m/s.
  - Optimized sensitivity minimum cost in sampled grid: 0.0700 at plane velocity 67.46 m/s and drop velocity 0.00 m/s.
  - Default sensitivity sampled cost range: 0.7800.
  - Optimized sensitivity sampled cost range: 0.8413.
  - The convergence curves are non-convex and irregular rather than smooth/parabolic, which supports using a GA instead of a local gradient method.

- Warnings / limitations:
  - The PDF bounds and notebook debug cell use a flow-rate entry near 8.21 m^3/s. Using 8.21e-6 releases zero tracked droplets with the provided super-particle scaling, so the assignment-consistent benchmark uses 8.21.
  - A static screenshot was saved instead of a full animation file to keep the workflow reliable and fast.
  - Optimized screenshot frame index: 454 with 46 visible tracked particles.