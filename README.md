ME144 / ME244 Project Workspace
===============================
<p align="center">
  <img src="https://github.com/user-attachments/assets/5ee9c3a4-f9eb-488b-b76a-6ed90f97a8a5" width="39%" />
  <img src="https://github.com/user-attachments/assets/bb90d0f7-a79a-4a81-800d-c590a7d16b49" width="60%" />
</p>

This repository contains Project 1 (optimization methods), Project 3 (swarm GA + physics simulation), Project 4 (geometry / dynamics / elastostatics coursework assets), Project X (3D emergent swarm control sandbox), and Project Y (a Chapter 4 inspired voxel/DEM hostile-flight drone study with GIF animation export).

Repository Layout
-----------------
- me144_toolbox/ : shared objectives, optimizers, simulation, and plotting utilities
- projects/project1/ : Project 1 scripts and outputs
- projects/project3/ : Project 3 swarm simulation + GA workflow and figures
- projects/project4/ : Project 4 notebooks and supporting geometry / dynamics / elastostatics folders
- projects/projectx/ : Project X (3D emergent control) scripts and docs
- projects/projecty/ : Project Y hostile-flight drone optimization, visuals, and docs

Requirements
------------
Create a virtual environment and install dependencies:

	python -m venv .venv
	.venv\Scripts\activate
	pip install -r requirements.txt

Project 1 (complete)
--------------------
Run Project 1 deliverables (Newton + GA):

	python -m projects.project1.run_project1
	python -m projects.project1.run_ga_project1
	python -m projects.project1.run_ga_phi_psi_project1

Outputs are saved under:
	projects/project1/output/figures/
	projects/project1/output/logs/

Project X (sandbox)
-------------------
Run the 3D emergent control pipeline:

	python projects\projectx\run_projectx_3d_animation.py

Implementation notes:
- The entry script is intentionally thin; most logic lives in:
	projects/projectx/projectx_anim/
	(config, environment generation, global state init, simulator, GA cost, pipeline, visualization)

Speed/debug options (environment variables):
- Fast smoke run:
	set PROJECTX_FAST=1
	python projects\projectx\run_projectx_3d_animation.py
- Skip plots/GIF (useful on slow machines):
	set PROJECTX_SKIP_VIZ=1
- Downsample animation frames (e.g., every 5th step):
	set PROJECTX_ANIM_STRIDE=5

Outputs are saved under:
	projects/projectx/output/figures/
	projects/projectx/output/logs/

Project 4
---------
Project 4 is organized as a notebook-driven workspace with supporting subfolders:

	projects/project4/main.ipynb
	projects/project4/geometry/
	projects/project4/dynamics/
	projects/project4/elastostatics/
	projects/project4/test_scripts/

Use the notebook as the main entry point and the subfolders for the supporting derivations, scripts, and experiments tied to the project.

Project Y (DEM drone hostile-flight study)
------------------------------------------
Run the Project Y optimization + simulation pipeline:

	python projects\projecty\run_projecty.py

Outputs are saved under:
	projects/projecty/output/

Implementation notes:
- The drone body is rendered as one sphere marker per active voxel on a regular lattice.
- The final animation and the design evolution animation are both exported as viewable GIFs in `projects/projecty/output/`.
- Additional documentation for Project Y is available under `projects/projecty/docs/`.

Project 3 (swarm GA + physics simulation)
----------------------------------------
Run the Project 3 notebook workflow from the Project 3 directory:

	cd projects\project3
	python write_parameters.py

Then open and run `test_scripts.ipynb` to generate the figures, MP4, and snapshot frames.

Outputs are saved under:
	projects/project3/figures/

Notes
-----
- Set random seeds in the scripts for reproducible runs.
- FFmpeg is optional but recommended for fast animation export (MP4/GIF).
