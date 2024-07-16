# Boundary Condition Enforcement with the Deep Ritz Method

There are a number of different techniques available to solve partial differential equations with solutions discretized by neural network. Two common methods are Physics-informed Networks (PINN's) and the Deep Ritz Method (DRM). These two methods construct different loss functions whose minimum corresponds to the PDE solution. The PDE and corresponding loss function for PINN's is

$$ \mathcal{L}(u) =f \rightarrow \int \Big( \mathcal{L}(u) - f \Big)^2 d\Omega$$ 

\noindent where $\mathcal{L}$ is a linear or nonlinear differential operator acting on the solution $u$ defined on the domain $\Omega$. Minimizing
