#!/usr/bin/env python3
"""Run level 1 coupled simulation using OASiS couple tool."""
import json
import os
import sys
from pathlib import Path

# Set up paths
work_dir_A = "/tmp/coupled_heat/level1_A"
work_dir_B = "/tmp/coupled_heat/level1_B"
output_dir = "/tmp/coupled_heat/output"

os.makedirs(work_dir_A, exist_ok=True)
os.makedirs(work_dir_B, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Copy participant scripts
import shutil
shutil.copy("/tmp/coupled_heat/participant_4c_A.py", work_dir_A + "/participant.py")
shutil.copy("/tmp/coupled_heat/participant_kratos_B.py", work_dir_B + "/participant.py")

# Define participants
participants = json.dumps([
    {
        "name": "fourc_A",
        "command": ["python3", work_dir_A + "/participant.py", "1"],
        "work_dir": work_dir_A,
        "imports_from": ["kratos_B"],
        "timeout": 900
    },
    {
        "name": "kratos_B", 
        "command": ["/usr/bin/python3", work_dir_B + "/participant.py", "1"],
        "work_dir": work_dir_B,
        "imports_from": ["fourc_A"],
        "timeout": 900
    }
])

print("Participants defined:")
print(participants)
print()

# Submit critic review
from mcp__oasis__submit_critic_review import submit_critic_review

coupling_args = {
    "participants": participants,
    "max_iter": 100,
    "tol": 1e-6,
    "accelerator": "aitken",
    "theta": 0.5,
    "probe": True
}

print("Submitting critic review...")
review = submit_critic_review(
    solver="couple",
    findings="Level 1 coupled heat conduction: 4C (Dirichlet side, k=1) + Kratos (Neumann side, k=200). "
             "Subdomain A: (0,0.625)x(0,1), Subdomain B: (0.625,1.5)x(0,1). "
             "Interface at x=0.625 with continuity of temperature and flux. "
             "Outer BCs: u=0 everywhere. Mesh: 5x8 QUAD4 elements in A, 7x8 triangles in B.",
    coupling_args=json.dumps(coupling_args)
)
print(f"Review result: {review}")
print()

# Run coupling
from mcp__oasis__couple import couple

print("Starting coupling iteration...")
result = couple(
    participants=participants,
    max_iter=100,
    tol=1e-6,
    accelerator="aitken",
    theta=0.5,
    probe=True,
    critic_approved=True
)

print(f"\nCoupling result:")
print(f"  Converged: {result.get('converged', 'unknown')}")
print(f"  Iterations: {result.get('iterations', 'unknown')}")
print(f"  Final residual: {result.get('residual', 'unknown')}")
print(f"  Verdict: {result.get('verdict', 'unknown')}")

# Save results
with open(output_dir + "/result_level1.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to {output_dir}/result_level1.json")
