# fluidx
Implementation of computer fluid dynamics (CFD) models solved by the finite element method (FEM) with dolfinx, the API of the [FEniCSx Project](https://fenicsproject.org/).

3D Modeling and meshing is done with [gmsh](https://gmsh.info/).

## Large eddy simulation (LES)
Obtained by applying a low pass filter to the Navier Stokes equations. Results in the filtered equations:

```math
\displaystyle\frac{\partial \bar{u}_i}{\partial x_i} = 0,
```
```math
\displaystyle\frac{\partial \bar{u}_i}{\partial t} + \displaystyle\frac{\partial}{\partial x_j}(\bar{u}_i\bar{u}_j) = -\displaystyle\frac{1}{\rho} \displaystyle\frac{\partial \bar{p}}{\partial x_i} - \displaystyle\frac{\partial \tau_{ij}}{\partial x_j} + \nu \displaystyle\frac{\partial^2 \bar{u}_i}{\partial x_j \partial x_j},
```

where $u_i$ is the i-th component of the velocity, $p$ is the pressure, and a bar over a variable means that it is filtered. $\tau_{ij}$ is the subgrid-scale stress (SGS) term, associated to the interaction between the length scales that are represented by the filtered equations and the scales that are lost with the filtering operation, and it needs to be modeled.

**Smagorinsky model**

In this model, the SGS term is modeled by the form

$$
\tau_{ij} - \displaystyle\frac{1}{3}\tau_{kk} \delta_{ij} = -2\nu_{\text{T}}\bar{S}_{ij}
$$
