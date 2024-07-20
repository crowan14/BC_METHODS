# Boundary Condition Enforcement with the Deep Ritz Method

There are a number of different techniques available to solve partial differential equations with solutions discretized by neural networks. Two common methods are Physics-informed Networks (PINN's) and the Deep Ritz Method (DRM). These two methods construct different loss functions whose minimum corresponds to the PDE solution. The PDE and corresponding loss function for PINN's is

$$ \mathcal{L}(u) =f \rightarrow \int \Big( \mathcal{L}(u) - f \Big)^2 d\Omega$$ 

where $\mathcal{L}$ is a linear or nonlinear differential operator acting on the solution $u$ defined on the domain $\Omega$. Minimizing the loss corresponds to solving the PDE at integration points. On the other hand, the Deep Ritz Method can only be used on PDE's with an associated variational "energy," a functional whose minimum corresponds to a solution. When $\Pi$ is such an energy functional, we have that

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

A zero temperature boundary condition will be applied to a semicircle centered at the origin with a radius of 1. The method of manufactured solutions can be used to obtain an analytical solution to compare approximate solutions against. The solution will be assumed to be 

$$ u(x_1,x_2) = a x_2(1-x_1^2-x_2^2) $$

which can be plugged into the governing PDE to compute the source term as

$$ f(x_1,x_2) = 8ax_1 $$

Note that the manufactured solution is zero along the semicircular boundary. This source term can be used in the approximate methods and we can compare the accuracy of the computed solution to the exact one. 

## Hard Boundary Enforcement

The solution is discretized with a neural network multiplied by a function which is zero along the boundary and non-zero inside the domain. This can be written as

$$ u(x_1,x_2) = g(x_1,x_2) \mathcal{N}(x_1,x_2;\theta) $$

where $g(x_1,x_2)$ enforces the boundary conditions and $\mathcal{N}$ is a neural network with parameters $\theta$. The Dirichlet boundary condition is enforced automatically with this method. The energy functional is not modified in any way because the boundary condition is built into the discretization. In the pytorch implementation, the zero temperature boundary condition along the straight side of the semi-circular domain will always be enforced in this way.

## Penalty Method

The solution is discretized such that only the straight-sided boundary is enforced automatically, and the energy functional is modified to add a penalty for violations of the boundary condition along the curved side of the semicircle. This reads

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega + p \int u(s)^2 ds $$

where $p>0$ is a penalty parameter and $s$ is a parameterization of the curved side of the domain.

## Self-Adaptive PINN's

The solution is discretized such that only the straight-sided boundary is enforced automatically, and the energy functional is modified to add a penalty for violations of the boundary condition along the curved side of the semicircle. Unlike the standard penalty formulation, there is a learnable penalty parameter associated with each integration point along the curved boundary. The loss is minimized over the parameters of the solution and maximized over the penalty parameters. The problem can be written as

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega + \int p(s) u(s)^2 ds $$

$$ \theta, p = min_{\theta} max_{p} \Pi  $$

## Nitsche's Method

Nitsche's method is a penalty method with an additional term added to the energy to preserve certain theoretical properties of the minimization problem. The solution is discretized such that the straight-sided boundary is enforced automatically. The discretization and loss function are

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega - \int \frac{\partial u}{\partial x_i} n_i u ds + p \int u(s)^2 ds $$

## Lagrange Multipliers

Lagrange multipliers are a standard technique for enforcing constraints and are relatively underexplored in the PINN's literature. The loss function is modified with a new unknown Lagrange multiplier field $\lambda$ that is used to enforce the constraint. A saddle point of the modified loss function (minimum over displacement parameters, maximum over Lagrange multiplier) corresponds to a minimum of the energy objective subject to the constraint implied by the Dirichlet boundaries. This reads

$$ u(x_1,x_2) = x_2 \mathcal{N}(x_1,x_2;\theta) $$

$$ \Pi = \int \frac{1}{2} \frac{\partial u}{\partial x_i} \frac{\partial u}{\partial x_i} -f u d\Omega + \int \lambda(s) u(s) ds  $$

## Augmented Lagrangian

The Augmented Lagrangian method is a kind of mid-point between Lagrange multiplier and penalty methods. The benefit of this method is that the penalty parameter may not need to be as large to accurately enforce the constraint. A sequence of problems is solved and the Lagrange multiplier and penalty parameter are updated at each step. The objective for the $k$-th problem is

$$ \Pi^k = \int \frac{1}{2} \frac{\partial u^k}{\partial x_i} \frac{\partial u^k}{\partial x_i} - f u^k d\Omega + \int \lambda^k u ds + \frac{1}{2}p^k \int u(s)^2 ds $$

The penalty $p^k$ is scheduled to increase by some predetermined factor from one step to the next. The Lagrange multiplier is updated with

$$ \lambda^{k+1}(s) = \lambda^k(s) + p^k u^k(s) $$

## Constrained Optimization

A standard constrained optimization method can be used such as Sequential Quadratic Programming (SQP) to enforce the Dirichlet boundary conditions. These methods use Lagrange multipliers under the hood, approximating the objective as quadratic at each point in the optimization process. We simply need to pass the energy objective to an out-of-the-box SQP method with a constraint saying that the displacement along the curved boundary is zero.

## Using the Code

Separate neural networks are introduced for each of the different methods. The Lagrange multiplier approach is implemented in three different ways: Lagrange multipliers stored at each integration point, the Lagrange multiplier discretized as a linear combination of shape functions, and the Lagrange multiplier discretized as a neural network. The code is separated into blocks that can be run separately after the neural networks are initialized, integration grids constructed, and some basic computations are performed (these can be run as blocks as well). Each method is split into its own block, and then a summary comparing all the methods is implemented at the end of the code.

## Note on Newton-type Optimization
Newton methods for constrained optimization do not do gradient descent, but instead solve the linear or nonlinear system expressing the condition for a stationary point of the objective. This means that they cannot distinguish between minima, maxima, and saddles given that all of these points have zero gradient. The MATLAB file "DRM_convexity.m" shows that Newton methods will sometimes find stationary points that are not minima for nonlinear discretizations such as those of a neural network. Thus, caution needs to be used when using optimization methods of this sort with the Deep Ritz Method. The eigenvalues of the Hessian matrix can be used to characterize whether the computed stationary point of the energy is a minimum, maximum or saddle. When the eigenvalues are all positive, the point is a minimum, when they are all negative, it is a maximum, and a mix of positive and negative values is a saddle point. A simple example is investigated in this file. We take the 1D elliptic energy functional with constant forcing

$$ \Pi = \int\frac{1}{2}\Big( \frac{\partial u}{\partial x} \Big)^2 - u dx  $$

and discretize the solution as a nonlinear function of degrees of freedom $a_1,\dots,a_N$. We use homogeneous Dirichlet boundaries on either side of the domain which are automatically satisfied by the discretization. The Hessian matrix can be shown to be


$$ H_{ij} = \int \frac{\partial^2 u}{\partial x \partial a_i} \frac{\partial^2 u}{\partial a_j} - \Big( \frac{\partial^2 u}{\partial x^2} + 1 \Big) \frac{\partial^2 u}{\partial a_i \partail a_j} dx $$
