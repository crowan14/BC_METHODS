# Boundary Condition Enforcement with the Deep Ritz Method

There are a number of different techniques available to solve partial differential equations with solutions discretized by neural network. Two common methods are Physics-informed Networks (PINN's) and the Deep Ritz Method (DRM). These two methods construct different loss functions whose minimum corresponds to the PDE solution. The PDE and corresponding loss function for PINN's is

$$ \mathcal{L}(u) =f \rightarrow \int \Big( \mathcal{L}(u) - f \Big)^2 d\Omega$$ 

where $\mathcal{L}$ is a linear or nonlinear differential operator acting on the solution $u$ defined on the domain $\Omega$. Minimizing the loss corresponds to accurately solving the PDE at integration points. On the other hand, the Deep Ritz Method can only be used on PDE's with an associated variational ``energy," a functional whose minimum corresponds to a solution. When $\Pi$ is such an energy functional, we have that

$$ \delta \Pi = 0 \rightarrow \mathcal{L}(u) = f $$

In other words, the PDE is the condition for the minimum of the energy functional. Thus an approximate PDE solution can be found by discretizing the energy and minimizing it. In this report, we focus on the Deep Ritz Method. Once a form of the loss has been chosen, it is necessary to find a way to enforce boundary conditions. Neumann boundary conditions are enforced weakly with the energy functional, but it is necessary to choose among a variety of techniques to enforce Dirichlet boundary conditions. Enforcing Dirichlet boundary conditions is often the most difficult part of solving PDE's with neural networks. We compare the following techniques for Dirichlet boundary enforcement in the context of the Deep Ritz Method:

1. Hard boundary enforcement
2. Penalty method
3. Self-Adaptive PINN's
4. Nitsche's Method
5. Lagrange Multipliers
6. Augmented Lagrangian Method
7. Constrained Optimization

These methods are briefly outlined below. The example problem will be 2D heat conduction on a semi-circular domain. The governing equation and energy functional are

$$ \frac{\partial^2 u}{\partial x_i \partial x_i}  + f = 0 $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega$$

A zero temperature boundary condition will be applied to a semicircle centered at the origin with a radius of 1. The method of manufactured solutions can be used to have an analytical solution to compare against. The solution will be assumed to be 

$$ u(x_1,x_2) = a x_2(1-x_1^2-x_2^2) $$

which can be plugged into the governing PDE to compute the source term as

$$ f(x_1,x_2) = 8ax_1 $$

Note that the manufactured solution is zero along the semicircular boundary. This source term can be used in the approximate methods and we can compare the accuracy of the computed solution to the exact one. 

## Hard Boundary Enforcement

The solution is discretized with a neural network multiplied by a function which is zero along the boundary and non-zero inside the domain. This can be written as

$$ u(x_1,x_2) = g(x_1,x_2) \mathcal{N}(x_1,x_2;\theta) $$

where $g(x_1,x_2)$ enforces the boundary conditions and $\mathcal{N}$ is a neural network with parameters $\theta$. The Dirichlet boundary condition is enforced automatically with this method. The energy funnctional is not modified in any way because the boundary condition is built into the discretization. In the pytorch implementation, the zero temperature boundary condition along the straight side of the semi-circular domain will always be enforced in this way.

## Penalty Method

The solution is discretized such that only the straight-sided boundary is enforced automatically, and the energy functional is modified to add a penalty for violations of the boundary condition along the curved side of the semicircle. This reads

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega + p \int u(s)^2 ds $$

where $p>0$ is a penalty parameter and $s$ is a parameterization of the curved side of the domain.

## Self-Adaptive PINN's

The solution is discretized such that only the straight-sided boundary is enforced automatically, and the energy functional is modified to add a penalty for violations of the boundary condition along the curved side of the semicircle. Unlike the standard penalty formulation, there is a learnable penalty parameter associated with each integration point along the curved boundary. The loss is minimized over the parameters of the solution and maximized over the penalty parameters. The problem can be written as

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega + \int p(s) u(s)^2 ds $$

$$ \theta, p = \text{argmin}_{\theta} \text{argmax}_{p} \Pi  $$

## Nitsche's Method

## Lagrange Multipliers

## Augmented Lagrangian

## Constrained Optimization
