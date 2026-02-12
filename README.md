ME144 / ME244 Project Workspace
===============================
![swarm_control_emergent](https://github.com/user-attachments/assets/5ee9c3a4-f9eb-488b-b76a-6ed90f97a8a5)

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

Notes
-----
- Set random seeds in the scripts for reproducible runs.
- FFmpeg is optional but recommended for fast animation export (MP4/GIF).
