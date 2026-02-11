ME144 / ME244 Project Workspace
===============================
![swarm_control_emergent](https://github.com/user-attachments/assets/bf6f00ca-2b65-4533-8bfe-4f5a4942547d)

This repository contains Project 1 (optimization methods) and Project X (3D emergent swarm control). Project X is a sandbox for Project 3 ideas; Project 1 is complete and polished.

Repository Layout
-----------------
- me144_toolbox/ : shared objectives, optimizers, simulation, and plotting utilities
- projects/project1/ : Project 1 scripts and outputs
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

Notes
-----
- Set random seeds in the scripts for reproducible runs.
- FFmpeg is optional but recommended for fast animation export (MP4/GIF).
