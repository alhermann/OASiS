#!/usr/bin/env python3
"""Simple Kratos test for 3D heat conduction"""
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

print("Creating model...")
model = KM.Model()
mp = model.CreateModelPart("domain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

# Add variables
for v in [KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX]:
    mp.AddNodalSolutionStepVariable(v)

print("Adding nodes...")
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, 1.0, 0.0, 0.0)
mp.CreateNewNode(3, 1.0, 1.0, 0.0)
mp.CreateNewNode(4, 0.0, 1.0, 0.0)
mp.CreateNewNode(5, 0.0, 0.0, 1.0)
mp.CreateNewNode(6, 1.0, 0.0, 1.0)
mp.CreateNewNode(7, 1.0, 1.0, 1.0)
mp.CreateNewNode(8, 0.0, 1.0, 1.0)

print(f"Nodes created: {len(list(mp.Nodes))}")

props = mp.CreateNewProperties(1)
for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, 1.0)
    node.SetSolutionStepValue(KM.TEMPERATURE, 0.0)

print("Creating element...")
mp.CreateNewElement("LaplacianElement3D8N", 1, [1,2,3,4,5,6,7,8], props)
print(f"Elements created: {len(list(mp.Elements))}")

print("Applying BCs...")
mp.Nodes[1].SetSolutionStepValue(KM.TEMPERATURE, 100.0)
mp.Nodes[1].Fix(KM.TEMPERATURE)

print("Adding DOFs...")
KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)

print("Setting up solver...")
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
strategy.Initialize()

print("Solving...")
strategy.Solve()

print("Done!")
for node in mp.Nodes:
    print(f"Node {node.Id}: T = {node.GetSolutionStepValue(KM.TEMPERATURE)}")
