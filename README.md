ME144 / ME244 Project Workspace
===============================
![swarm_control_emergent](https://github.com/user-attachments/assets/bf6f00ca-2b65-4533-8bfe-4f5a4942547d)

This repository contains Project 1 (optimization methods), Project 3 (swarm GA + physics simulation), and Project X (3D emergent swarm control sandbox).

Repository Layout
-----------------
- me144_toolbox/ : shared objectives, optimizers, simulation, and plotting utilities
- projects/project1/ : Project 1 scripts and outputs
- projects/project3/ : Project 3 swarm simulation + GA workflow and figures
- projects/projectx/ : Project X (3D emergent control) scripts and docs

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

Outputs are saved under:
	projects/projectx/output/figures/
	projects/projectx/output/logs/

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
