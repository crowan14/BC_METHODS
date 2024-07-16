# Boundary Condition Enforcement with the Deep Ritz Method

There are a number of different techniques available to solve partial differential equations with solutions discretized by neural network. Two common methods are Physics-informed Networks (PINN's) and the Deep Ritz Method (DRM). These two methods construct different loss functions whose minimum corresponds to the PDE solution. The loss function for PINN's is

$$ \int \Big( \mathcal{L}(u) - f \Big)^2 d\Omega$$ 
