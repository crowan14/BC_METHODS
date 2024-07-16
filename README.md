# Boundary Condition Enforcement with the Deep Ritz Method

There are a number of different techniques available to solve partial differential equations with solutions discretized by neural network. Two common methods are Physics-informed Networks (PINN's) and the Deep Ritz Method (DRM). These two methods construct different loss functions whose minimum corresponds to the PDE solution. The PDE and corresponding loss function for PINN's is

$$ \mathcal{L}(u) =f \rightarrow \int \Big( \mathcal{L}(u) - f \Big)^2 d\Omega$$ 

where $\mathcal{L}$ is a linear or nonlinear differential operator acting on the solution $u$ defined on the domain $\Omega$. Minimizing the loss corresponds to accurately solving the PDE at integration points. On the other hand, the Deep Ritz Method can only be used on PDE's with an associated variational ``energy," a functional whose minimum corresponds to a solution. When $\Pi$ is such an energy functional, we have that

$$ \delta \Pi = 0 \rightarrow \mathcal{L}(u) = f $$

In other words, the PDE is the condition for the minimum of the energy functional. Thus an approximate PDE solution can be found by discretizing the energy and minimizing it. In this report, we focus on the Deep Ritz Method. Once a form of the loss has been chosen, it is necessary to find a way to enforce boundary conditions. Neumann boundary conditions are enforced weakly with the energy functional, but it is necessary to choose among a variety of techniques to enforce Dirichlet boundary conditions. Enforcing Dirichlet boundary conditions is often the most difficult part of solving PDE's with neural networks. We compare the following techniques for Dirichlet boundary enforcement in the context of the Deep Ritz Method:

1. Hard boundary enforcement
2. Penalty method
3. Self-Adaptive PINN's
4. Variants of Lagrange Multipliers
5. Augmented Lagrangian Method
6. Constrained Optimization

## Hard Boundary Enforcement

## Penalty Method

## Self-Adaptive PINN's

## Lagrange Multipliers

## Augmented Lagrangian

## Constrained Optimization
