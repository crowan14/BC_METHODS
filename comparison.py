import torch
import numpy as np
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Callable, Iterator, Union
import scipy.optimize
from scipy.optimize import OptimizeResult
import time

# %matplotlib qt

######################################################

# DEEP RITZ METHOD

# COMPARE TECHNIQUES FOR ENFORCEMENT OF BC'S: HARD ENFORCEMENT WITH MULTIPLICATION (A), STANDARD PENALTY APPROACH (B), 
# SA-PINN PENALTY (C), DISCRETE LAGRANGE MULTIPLIER (D), CONTINUOUS LAGRANGE MULTIPLIER (E), NITSCHE'S METHOD (F),
# AUGMENTED LAGRANGIAN (G), NEURAL NETWORK LAGRANGE MULTIPLIER (H), AND SQP CONSTRAINT ENFORCEMENT (I)

# TEST ON 2D HEAT CONDUCTION  WITH CONSTANT CONDUCTIVITY OF 1

# USE MANUFACTURED SOLUTION TO DETERMINE HEAT SOURCE

# SOLUTION DISCRETIZED WITH TWO LAYER NEURAL NETWORK

# ZERO TEMPERATURE BOUNDARY CONDITION FOR SEMI-CIRCLE DOMAIN:

#                 (0,1)
#                 --x--
#                   | 
#         /         |        \
#                   |    
#        |          |         |
# (-1,0) x--------------------x (1,0)

######################################################

#shape of domain
def semicircle(x1):
    x2 = np.power( 1-np.power(x1,2) , 0.5 )
    return x2

#slope of semicircular boundary
def d_semicircle(x1):
    slope = - x1 / ( 1-x1**2 )**(0.5)
    return slope

#manufactured solution
a = 2
def u(x):
    val = a * x[:,1] * ( 1 - x[:,0]**2 - x[:,1]**2 )
    return val

#heat source
def f(x):
    val = 8 * a * x[:,1]
    return val

#width of hidden layers
n = 10

######################################################

#%% DEFINE NEURAL NETWORKS FOR DIFFERENT METHODS

#NETWORK FOR HARD BOUNDARY ENFORCEMENT
class SolutionA(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #shape functions have BC built in
        x1 = torch.reshape( x[:,0] , (len(x),1) )
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = torch.mul( x2 , -x1**2 - x2**2 + 1 )
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
######################################################

#NETWORK FOR STANDARD PENALTY APPROACH
class SolutionB(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #build in straight sided boundary condition to shape functions
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]

    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
######################################################

#NETWORKS FOR SELF-ADAPTIVE PINN'S 
class SolutionC(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #build in straight sided boundary condition to shape functions
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]

    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
class SA(nn.Module):
    def __init__( self , pts ):
        super().__init__()
        
        vec = 1*torch.rand(pts)
        self.layer1 = nn.Parameter(vec)
        
    def forward( self ):
          
        y = self.layer1
        
        return y
    
######################################################
    
#NETWORKS FOR DISCRETE LAGRANGE MULTIPLIER
class SolutionD(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #build in straight sided boundary condition to shape functions
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]

    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
class LagrangeD(nn.Module):
    def __init__( self , pts ):
        super().__init__()
        
        vec = 5*torch.rand(pts)
        self.layer1 = nn.Parameter(vec)
        
    def forward( self ):
          
        y = self.layer1
        
        return y

######################################################

#NETWORKS FOR CONTINUOUS LAGRANGE MULTIPLIER
class SolutionE(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #build in straight sided boundary condition to shape functions
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]

    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
class LagrangeE(nn.Module):
    def __init__( self , N ):
        super().__init__()
        
        vec = 5*torch.rand(N)
        self.layer1 = nn.Parameter(vec)
        
    def forward( self ):
          
        y = self.layer1
        
        return y
    
######################################################

#NETWORK FOR NITSCHE'S METHOD
class SolutionF(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #shape functions have straight sided BC built in
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
######################################################

#NETWORK FOR AUGMENTED LAGRANGIAN
class SolutionG(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #shape functions have straight sided BC built in
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi

######################################################

#NETWORKS FOR NN LAGRANGE MULTIPLIER
class SolutionH(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #shape functions have straight sided BC built in
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
class LagrangeH(nn.Module):
    def __init__( self , n ):
        super().__init__()
        
        #define layers and activation
        self.layer_1 = nn.Linear( 1 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        self.shift = nn.Parameter(2*torch.rand(1))
        
        self.act = nn.Tanh()
        
    def forward( self , x ):
          
        #two hidden layer network
        y = self.layer_1(x)
        # y = self.act(y)
        y = torch.sin(y)
        y = self.layer_2(y)
        #y = self.act(y) 
        y = torch.sin(y)
        y = self.output(y)
        y = y + self.shift
        
        return y

######################################################

#NETWORK FOR SQP CONSTRAINED OPTIMIZATION
class SolutionI(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
        #define layers and activation
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        
        #shape functions have straight sided BC built in
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        g = x2
        y = torch.mul( g , y )
        
        y = self.output(y)
        
        return y
  
    def gradient( self , x ):
        
        #get spatial gradient of solution at integration points
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy_density( self , x ):
        
        #get solution and spatial gradient at integration points
        Eval = self.gradient(x)
        u = Eval[0]
        grad_u = Eval[1]
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        #form variational energy for specified heat source
        Pi = 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - torch.mul( f(x).unsqueeze(1) , u ) 
        
        return Pi
    
#%%

######################################################

#%% GRIDS FOR INTEGRATION AND PLOTTING

#uniform grid along x1 axis
pts = 50
x1 = np.linspace(-1,1,pts)
dx1 = x1[1]-x1[0]
x1 = x1 + dx1/2
x1 = x1[:len(x1)-1]

#1d grid
x1_grid = torch.tensor( np.atleast_2d(x1).T , dtype=torch.float32 )

#interior integration grid
grid = []
for i in range(pts-1):
    x2 = dx1/2
    while x2 <= semicircle( x1[i] ):
        grid += [  [ x1[i] , x2  ]  ]
        x2 += dx1
x_grid = torch.tensor( grid , dtype=torch.float32 )

#boundary grid
bc_grid = np.zeros((pts-1,2))
bc_grid[:,0] = x1
bc_grid[:,1] = semicircle(x1)
bc_grid = torch.tensor( bc_grid , dtype=torch.float32 )

#boundary integration weights
ds = np.zeros((len(x1),1))
for i in range(len(x1)):
    ds[i] = dx1 * ( 1 + d_semicircle( x1[i] )**2 )**0.5
ds = torch.tensor( ds , dtype=torch.float32 )

#visualize integration grids
plt.figure()

plt.subplot(2,1,1)
plt.scatter( x_grid[:,0] , x_grid[:,1] , color='b' , label='Interior Integration Grid' )
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.subplot(2,1,2)
plt.scatter( bc_grid[:,0] , bc_grid[:,1] , color='r' , label='Boundary Integration Grid' ) 
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()

plt.show()

#grid for plotting solution
scale = 5
X1 , X2 = np.meshgrid( np.linspace(-1,1,scale*pts) , np.linspace(0,1,scale*pts) )
col1 = np.reshape( X1 , ((scale*pts)**2,) ) 
col2 = np.reshape( X2 , ((scale*pts)**2,) ) 
plot_grid = np.zeros(((scale*pts)**2,2))
plot_grid[:,0] = col1
plot_grid[:,1] = col2
plot_grid = torch.tensor( plot_grid , dtype=torch.float32 )

xx = np.linspace(-1,1,scale*pts)

#%%

######################################################

#%%COMPUTE SOME THINGS

#compute normal vector to curved boundary
dcdx1 =  np.asarray( [ d_semicircle(elem) for elem in x1 ] )
normal = np.zeros((pts-1,2))
normal[:,0] = -dcdx1
normal[:,1] = 1
mags = np.power( np.power(normal[:,0],2) + np.power(normal[:,1],2) , 0.5 ) 
normal[:,0] = np.divide( normal[:,0] , mags)
normal[:,1] = np.divide( normal[:,1] , mags)
normal = torch.tensor( normal , dtype=torch.float32 )

#construct exact solution
exact_interior = u(x_grid).unsqueeze(1)

#plot exact solution
sol = u(plot_grid)
Zex = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZex = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZex[i,j] = Zex[i,j]
        else:
            maskedZex[i,j] = np.nan
            

#compute energy of exact solution with energy density
def pi(x):
    val = 0.5 * ( ( -2*a*x[:,0]*x[:,1] )**2 + ( a - a*x[:,0]**2 - 3*a*x[:,1]**2 )**2 ) - torch.mul( f(x) , u(x) )
    return val

energy = float( torch.sum( dx1**2 * pi(x_grid) ) )

#number of parameters in solution neural network
model = SolutionA(n)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#%%

######################################################

#%% WRAPPER CODE FOR SCIPY SQP OPTIMIZATION

def ravel_pack(tensors):
    # Faster version of nn.utils.parameters_to_vec, modified slightly from
    # https://github.com/gngdb/pytorch-minimize/blob/master/pytorch_minimize/optim.py
    def numpyify(tensor):
        if tensor is None:
            return np.array([0.0])
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    x = np.concatenate([numpyify(tensor).ravel() for tensor in tensors], 0)
    # cast to float64
    # x = x.astype(np.float64)
    return x


def ravel_pack_float64(tensors):
    # Faster version of nn.utils.parameters_to_vec, modified slightly from
    # https://github.com/gngdb/pytorch-minimize/blob/master/pytorch_minimize/optim.py
    def numpyify(tensor):
        if tensor is None:
            return np.array([0.0])
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    x = np.concatenate([numpyify(tensor).ravel() for tensor in tensors], 0)
    # cast to float64
    x = x.astype(np.float64)
    return x

def torch_to_np(x: Union[torch.Tensor, np.array, float]):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


class TorchScipyOptimizer(torch.optim.Optimizer):
    """
    Takes PyTorch stuff for constrained optimization and gives it to scipy.optimize.minimize
    Altogether REPLACES the traditional training loop. Extends Torch Optimizer class.
    Most of the computational cost will be dealing with gradient things: propagating via AD,
    grabbing the parameter grad attributes, flattening to vector, then converting to numpy.

    Should support using a single GPU in typical PyTorch fashion, but the constant switching
    to GPU for PyTorch and back to CPU for Numpy is quite expensive. Maybe worth it for very
    expensive PyTorch functions that greatly benefit from GPU acceleration.

    Like https://github.com/gngdb/pytorch-minimize/blob/master/pytorch_minimize/optim.py#L99,
    but for my needs (constrained, one method).
    """

    def __init__(self, parameters: Iterator[torch.Tensor],
                 minimizer_args: dict = {'maxiter': 100},
                 callback: Callable[[OptimizeResult], None] = lambda *args: None):
        """Initialize the settings passed to the optimizer.

        Args:
            parameters: iterator of torch.Tensor
            minimizer_args: dict of arguments to pass to scipy.optimize.minimize,
                (default {'maxiter':100})
            callback: function to call after each iteration of scipy.optimize.minimize,
                which should take the argument intermediate_result: OptimizeResult,
                (default lambda *args: None),
        """

        super().__init__(parameters, defaults=minimizer_args)
        self.callback = callback

    def step(self,
             obj_fn: Callable[[], torch.Tensor],
             con_fn: Callable[[], torch.Tensor],
             lower_bnd: Union[torch.Tensor, np.array, float] = -np.inf,
             upper_bnd: Union[torch.Tensor, np.array, float] = 0.0) -> OptimizeResult:
        """Minimize obj_fn subject to con_fn <= 0. Uses settings from initialization.

        Constrained optimization currently only using trust-constr method. Handles
        computation of objective gradient and constraint Jacobian via PyTorch's AD.
        Casts these to numpy arrays for scipy.optimize.minimize for each step. The
        Jacobian calculation calls the constraint function once and calls backward
        for each constraint.

        Args:
            obj_fn: function that returns a scalar tensor
            con_fn: function that returns a vector tensor of size (n_constraints, 1)
            lower_bnd: lower bound for constraints, defaults to -inf
            upper_bnd: upper bound for constraints, defaults to 0.0

        Returns:
            scipy.optimize.OptimizeResult
        """

        assert len(self.param_groups) == 1, "Only one parameter group is supported"
        parameters = self.param_groups[0]['params']

        x0 = ravel_pack(parameters)
        # get sizes for constraints and parameter sizes
        n_con = con_fn().numel()
        n_params = sum([p.numel() for p in parameters])

        np_lower_bnd = torch_to_np(lower_bnd)
        np_upper_bnd = torch_to_np(upper_bnd)

        def vec_to_params(x):
            nn.utils.vector_to_parameters(torch.tensor(x, dtype=torch.float32,
                                                       device=parameters[0].device), parameters)

        def np_obj_fun(x):
            vec_to_params(x)
            obj = obj_fn()
            return obj.item()

        def np_obj_jac(x):
            # Possibly redundant objective call. Could use obj.backward(), rather than redoing this...
            vec_to_params(x)

            self.zero_grad()
            obj = obj_fn()
            obj.backward()

            # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            grads = ravel_pack([p.grad for p in parameters])
            return grads

        def np_ineq_fun(x):
            vec_to_params(x)
            con = con_fn()
            # if con.device != torch.device('cpu'):
            #     con = con.cpu()
            # return con.detach().numpy()[:,0]
            return con.view(-1).cpu().detach().numpy()

        def np_ineq_jac(x):
            # This likely isn't efficient, and was written for PyTorch before functorch was added.
            vec_to_params(x)

            jac = np.zeros((n_con, n_params), dtype=np.float32)
            con = con_fn()
            # eeem = torch.zeros((n_con,1), dtype=torch.float32, device=con.device)
            eeem = torch.zeros_like(con)
            for i in range(n_con):
                self.zero_grad()
                eeem[i] = 1.0
                con.backward(eeem, retain_graph=True)
                eeem[i] = 0.0

                # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
                for p in parameters:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)

                grads = ravel_pack([p.grad for p in parameters])
                jac[i, :] = grads

            return jac

        nonlinear_constraint = scipy.optimize.NonlinearConstraint(np_ineq_fun, np_lower_bnd, np_upper_bnd,
                                                                  jac=np_ineq_jac, hess=scipy.optimize.BFGS())

        res = scipy.optimize.minimize(np_obj_fun, x0, jac=np_obj_jac,
                                      constraints=nonlinear_constraint,
                                      hess=scipy.optimize.BFGS(),
                                      method='trust-constr',
                                      options=self.defaults,
                                      callback=self.callback)

        return res


class TorchScipyOptimizer_SLSQP(torch.optim.Optimizer):
    """
    See above. Only changed handling of constraints (and grad) for SLSQP method.
    """

    def __init__(self, parameters: Iterator[torch.Tensor],
                 minimizer_args: dict = {'maxiter': 100},
                 callback: Callable[[OptimizeResult], None] = lambda *args: None):
        """Initialize the settings passed to the optimizer.

        Args:
            parameters: iterator of torch.Tensor
            minimizer_args: dict of arguments to pass to scipy.optimize.minimize,
                (default {'maxiter':100})
            callback: function to call after each iteration of scipy.optimize.minimize,
                which should take the argument intermediate_result: OptimizeResult,
                (default lambda *args: None),
        """

        super().__init__(parameters, defaults=minimizer_args)
        self.callback = callback

    def step(self,
             obj_fn: Callable[[], torch.Tensor],
             con_fn: Callable[[], torch.Tensor]) -> OptimizeResult:
        """Minimize obj_fn subject to con_fn == 0. Uses settings from initialization.

        Args:
            obj_fn: function that returns a scalar tensor
            con_fn: function that returns a vector tensor of size (n_constraints, 1)

        Returns:
            scipy.optimize.OptimizeResult
        """

        assert len(self.param_groups) == 1, "Only one parameter group is supported"
        parameters = self.param_groups[0]['params']

        x0 = ravel_pack_float64(parameters)
        # get sizes for constraints and parameter sizes
        n_con = con_fn().numel()
        n_params = sum([p.numel() for p in parameters])

        def vec_to_params(x):
            nn.utils.vector_to_parameters(torch.tensor(x, dtype=torch.float32,
                                                       device=parameters[0].device), parameters)

        def np_obj_fun(x):
            vec_to_params(x)
            obj = obj_fn()
            return obj.item()

        def np_obj_jac(x):
            # Possibly redundant objective call. Could use obj.backward(), rather than redoing this...
            vec_to_params(x)

            self.zero_grad()
            obj = obj_fn()
            obj.backward()

            # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            grads = ravel_pack_float64([p.grad for p in parameters])
            return grads

        def np_ineq_fun(x):
            vec_to_params(x)
            con = con_fn()
            # if con.device != torch.device('cpu'):
            #     con = con.cpu()
            # return con.detach().numpy()[:,0]
            return con.view(-1).cpu().detach().numpy()

        def np_ineq_jac_general(x, i):
            # np_ineq_jac, but for constraint i. This has redundant forward passes of con_fn...
            vec_to_params(x)

            self.zero_grad()
            con = con_fn()
            eeem = torch.zeros_like(con)
            eeem[i] = 1.0
            con.backward(eeem, retain_graph=True)

            # Iterate through p.grad. If it is none, replace it with zeros of the appropriate size
            for p in parameters:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            grads = ravel_pack_float64([p.grad for p in parameters])

            return grads

        # Constraints for COBYLA, SLSQP are defined as a list of dictionaries. Each dictionary with fields:
        # typestr
        # Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
        # funcallable
        # The function defining the constraint.
        # jaccallable, optional
        # The Jacobian of fun (only for SLSQP).
        # argssequence, optional
        # Extra arguments to be passed to the function and Jacobian.
        list_nonlinear_constraint = [{}] * n_con
        for i in range(n_con):
            list_nonlinear_constraint[i] = {'type': 'eq',
                                       'fun': lambda x: np_ineq_fun(x)[i],
                                       'jac': lambda x: np_ineq_jac_general(x,i)}


        res = scipy.optimize.minimize(np_obj_fun, x0, jac=np_obj_jac,
                                      constraints=list_nonlinear_constraint,
                                      method='SLSQP',
                                      options=self.defaults,
                                      callback=self.callback)

        return res



class ConstrainedOptimizerTracker():
    def __init__(self, maxiter, n_constraints, obj, con, lb=-np.inf, ub=0.0):
        self.con_history = torch.zeros((maxiter, n_constraints))
        self.obj_history = torch.zeros((maxiter, 1))
        self.obj = obj
        self.con = con
        self.lb = lb
        self.ub = ub
        self.current_iter = 0
        self.current_obj = None
        self.current_con = None
        self.count = 1

    def constraint_violation(self, con):
        # account for lb and ub, find total constraint violation lb <= con <= ub
        ub_viol = torch.relu(con - self.ub)
        lb_viol = torch.relu(self.lb - con)
        return ub_viol + lb_viol

    def callback(self, intermediate_result):
        # if self.current_obj is None:
        #     self.current_obj = self.obj_fn()
        # if self.current_con is None:
        #     self.current_con = self.con_fn()

        self.obj_history[self.current_iter, :] = self.current_obj.detach()
        self.con_history[self.current_iter, :] = self.current_con.detach().flatten()
        self.current_iter += 1
        con_norm = torch.sum(self.constraint_violation(self.current_con))
        #print(f'Iteration {self.current_iter}, Objective: {self.current_obj.item()}, Constraint Violation: {con_norm}')
        self.count += 1
        
    def obj_fn(self):
        self.current_obj = self.obj()
        return self.current_obj

    def con_fn(self):
        self.current_con = self.con()
        return self.current_con

    def show_history(self):
        n_quantiles = 101
        quantiles = torch.linspace(0, 1, n_quantiles)
        # plot symmetrically
        con_quants = torch.quantile(self.con_history, quantiles, axis=1)
        med = torch.median(self.con_history, dim = 1)
        min = torch.min(self.con_history, dim = 1)
        max = torch.max(self.con_history, dim = 1)
        epoch_range = range(self.current_iter)

        # get percent of constraints that are violated
        viol = self.constraint_violation(self.con_history)
        n_violated = torch.sum(viol > 0, dim=1)
        percent_violated = n_violated / self.con_history.shape[1]

        fig, ax = plt.subplots(3,1, figsize=(6,12))
        ax[0].plot(epoch_range, self.obj_history[:self.current_iter])
        ax[0].set_ylabel('Objective History')
        ax[0].set_yscale('log')

        # use quantiles to fill between symmetrically
        for quant_idx in range(n_quantiles//2):
            ax[1].fill_between(epoch_range,
                               con_quants[quant_idx, :self.current_iter], con_quants[-quant_idx-1, :self.current_iter],
                               alpha=1.0 / (n_quantiles//2), color='blue', edgecolor=None)

        # plot median as dotted line
        ax[1].plot(epoch_range, med.values[:self.current_iter], linestyle='--', color='blue', alpha=0.5)
        ax[1].plot(epoch_range, min.values[:self.current_iter], linestyle=':', color='blue', alpha=0.5)
        ax[1].plot(epoch_range, max.values[:self.current_iter], linestyle=':', color='blue', alpha=0.5)
        # plot ub and lb, if they aren't inf or -inf
        if self.lb > -np.inf:
            ax[1].axhline(self.lb, color='k', linestyle='--', alpha=0.5)
        if self.ub < np.inf:
            ax[1].axhline(self.ub, color='k', linestyle='--', alpha=0.5)
        ax[1].set_ylabel('Constraint History')
        ax[1].set_yscale('symlog')

        ax[2].plot(epoch_range, percent_violated[:self.current_iter])
        ax[2].set_ylabel('Percent of Constraints Violated')

        # layout, label bottom x
        plt.tight_layout()
        ax[2].set_xlabel('Iteration')
        return fig, ax

#%%

######################################################

#%% TRAIN A -- HARD BOUNDARY ENFORCEMENT

#initialize network
uA = SolutionA(n)

#training parameters
epochsA = 5000
lrA = 1e-3
lossesA = np.zeros(epochsA)

#ADAM optimization
optimizer_sol = torch.optim.Adam( uA.parameters() , lr=lrA )

t0 = time.time()
print('begin training A...')
for i in range(epochsA):
    
    optimizer_sol.zero_grad()
    
    bulk = torch.sum( dx1**2 * uA.energy_density( x_grid ) )
    loss = bulk
    
    loss.backward()
    
    optimizer_sol.step()  
    
    lossesA[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesA[i]}')
    
t1 = time.time()

#run time
tA = t1-t0

#error
mseA = torch.sum( (exact_interior - uA.forward(x_grid))**2 ) / len(x_grid)
b_errorA = torch.sum( uA.forward( bc_grid )**2 ) / len(bc_grid)

mseA = mseA.detach().numpy()
b_errorA = b_errorA.detach().numpy()
        
#converged solution
sol = uA.forward(plot_grid).detach().numpy()
ZA = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZA = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZA[i,j] = ZA[i,j]
        else:
            maskedZA[i,j] = np.nan
    
#l2 error over domain
l2_errorA = ( maskedZA-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorA , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uA.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%
            
######################################################

#%% TRAIN B -- STANDARD PENALTY ENFORCEMENT  

#initialize network
uB = SolutionB(n)

#penalty parameter
penB = 5E3

#training parameters
epochsB = 40000
lrB = 1e-3
lossesB = np.zeros(epochsB)

#ADAM optimization
optimizer_sol = torch.optim.Adam( uB.parameters() , lr=lrB )

t0 = time.time()
print('begin training B...')
for i in range(epochsB):
    
    optimizer_sol.zero_grad()
    
    bulk = torch.sum( dx1**2 * uB.energy_density( x_grid ) )
    penalty = penB * torch.sum( torch.mul( uB.forward( bc_grid )**2 , ds ) )
    loss = bulk + penalty
    
    loss.backward()
    
    optimizer_sol.step()  
    
    lossesB[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesB[i]}')   
  
t1 = time.time()

#run time
tB = t1-t0

#error
mseB = torch.sum( (exact_interior - uB.forward(x_grid))**2 ) / len(x_grid)
b_errorB = torch.sum( uB.forward( bc_grid )**2 ) / len(bc_grid)

mseB = mseB.detach().numpy()
b_errorB = b_errorB.detach().numpy()
        
#converged solution
sol = uB.forward(plot_grid).detach().numpy()
ZB = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZB = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZB[i,j] = ZB[i,j]
        else:
            maskedZB[i,j] = np.nan
    
#l2 error over domain
l2_errorB = ( maskedZB-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorB , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uB.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()
            
#%%

######################################################

#%% TRAIN C -- SELF-ADAPTIVE PINN'S

#initialize networks
uC = SolutionC(n)
penC = SA(pts-1)

#training parameters
epochsC = 30000
lrC = 1e-3
lossesC = np.zeros(epochsC)

#ADAM optimization for both, maximization over penalty parameters
optimizer_sol = torch.optim.Adam( uC.parameters() , lr=lrC )
optimizer_pen = torch.optim.Adam( penC.parameters() , lr=100*lrC , maximize=True )

t0 = time.time()
print('begin training C...')
for i in range(epochsC):
    
    optimizer_sol.zero_grad()
    optimizer_pen.zero_grad()
    
    bulk = torch.sum( dx1**2 * uC.energy_density( x_grid ) )
    penalty = torch.sum( torch.mul( torch.mul( uC(bc_grid)**2 , penC.forward().unsqueeze(1) ) , ds ) )
    loss = bulk + penalty
    
    loss.backward()
    
    optimizer_sol.step()
    optimizer_pen.step()
    
    
    lossesC[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesC[i]}')
 
t1 = time.time()

#run time
tC = t1-t0   
     
#error
mseC = torch.sum( (exact_interior - uC.forward(x_grid))**2 ) / len(x_grid)
b_errorC = torch.sum( uC.forward( bc_grid )**2 ) / len(bc_grid)

mseC = mseC.detach().numpy()
b_errorC = b_errorC.detach().numpy()
        
#converged solution
sol = uC.forward(plot_grid).detach().numpy()
ZC = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZC = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZC[i,j] = ZC[i,j]
        else:
            maskedZC[i,j] = np.nan
    
#l2 error over domain
l2_errorC = ( maskedZC-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorC , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uC.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()
            
#%%

######################################################

#%% TRAIN D -- DISCRETE LAGRANGE MULTIPLIER

#initialize networks
uD = SolutionD(n)
lamD = LagrangeD(pts-1)

#training parameters
epochsD = 30000
lrD = 5e-5
lossesD = np.zeros(epochsD)

#ADAM optimization for both, maximization over penalty parameters
optimizer_sol = torch.optim.Adam( uD.parameters() , lr=lrD )
optimizer_lam = torch.optim.Adam( lamD.parameters() , lr=10*lrD , maximize=True )

t0 = time.time()
print('begin training D...')
for i in range(epochsD):
    
    optimizer_sol.zero_grad()
    optimizer_lam.zero_grad()
    
    bulk = torch.sum( dx1**2 * uD.energy_density( x_grid ) )
    lagrange = torch.sum( torch.mul( torch.mul( uD(bc_grid) , lamD.forward().unsqueeze(1) ) , ds ) )
    loss = bulk + lagrange
    
    loss.backward()
    
    optimizer_sol.step()
    optimizer_lam.step()
    
    
    lossesD[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesD[i]}')
        
t1 = time.time()

#run time
tD = t1-t0
        
#error
mseD = torch.sum( (exact_interior - uD.forward(x_grid))**2 ) / len(x_grid)
b_errorD = torch.sum( uD.forward( bc_grid )**2 ) / len(bc_grid)

mseD = mseD.detach().numpy()
b_errorD = b_errorD.detach().numpy()
        
#converged solution
sol = uD.forward(plot_grid).detach().numpy()
ZD = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZD = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZD[i,j] = ZD[i,j]
        else:
            maskedZD[i,j] = np.nan
    
#l2 error over domain
l2_errorD = ( maskedZD-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorD , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uD.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()
            
#%%

######################################################

#%% TRAIN E -- CONTINUOUS LAGRANGE MULTIPLIER

#number of shape functions
N = 20

#initialize networks
uE = SolutionE(n)
lamE = LagrangeE(N)

#training parameters
epochsE = 30000
lrE = 5e-5
lossesE = np.zeros(epochsE)

#ADAM optimization for both, maximization over penalty parameters
optimizer_sol = torch.optim.Adam( uE.parameters() , lr=lrE )
optimizer_lam = torch.optim.Adam( lamE.parameters() , lr=10*lrE , maximize=True )

shapes = torch.zeros((pts-1,N))
for i in range(N):
    shapes[:,i] = ( torch.cos( i*np.pi*x1_grid ) / (i+1) ).squeeze()

t0 = time.time()
print('begin training E...')
for i in range(epochsE):
    
    optimizer_sol.zero_grad()
    optimizer_lam.zero_grad()
    
    bulk = torch.sum( dx1**2 * uE.energy_density( x_grid ) )
    Lambda = torch.matmul( shapes , lamE.forward().unsqueeze(1) )
    lagrange = torch.sum( torch.mul( torch.mul( uE(bc_grid) , Lambda ) , ds ) )
    loss = bulk + lagrange
    
    loss.backward()
    
    optimizer_sol.step()
    optimizer_lam.step()
    
    lossesE[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesE[i]}')
        
t1 = time.time()

#run time
tE = t1-t0
    
#error
mseE = torch.sum( (exact_interior - uE.forward(x_grid))**2 ) / len(x_grid)
b_errorE = torch.sum( uE.forward( bc_grid )**2 ) / len(bc_grid)

mseE = mseE.detach().numpy()
b_errorE = b_errorE.detach().numpy()
        
#converged solution
sol = uE.forward(plot_grid).detach().numpy()
ZE = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZE = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZE[i,j] = ZE[i,j]
        else:
            maskedZE[i,j] = np.nan
    
#l2 error over domain
l2_errorE = ( maskedZE-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorE , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uE.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%

######################################################

#%% TRAIN F -- NITSCHE'S METHOD

#initialize network
uF = SolutionF(n)

#penalty parameter
penF = 1E3

#training parameters
epochsF = 20000
lrF = 1e-3
lossesF = np.zeros(epochsF)

#ADAM optimization
optimizer_sol = torch.optim.Adam( uF.parameters() , lr=lrF )

t0 = time.time()
print('begin training F...')
for i in range(epochsF):
    
    optimizer_sol.zero_grad()
    
    bulk = torch.sum( dx1**2 * uF.energy_density( x_grid ) )
    penalty = penF * torch.sum( torch.mul( uF.forward( bc_grid )**2 , ds ) )
    
    grad = uF.gradient(bc_grid)[1]
    normal_grad = torch.sum( torch.mul( grad , normal ) , 1 ).unsqueeze(1)
    nitsche = torch.sum( torch.mul (  normal_grad * u( bc_grid ).unsqueeze(1) , ds ) )
    
    loss = bulk - nitsche + penalty
    
    loss.backward()
    
    optimizer_sol.step()  
    
    lossesF[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesF[i]}')   
        
t1 = time.time()

#run time
tF = t1-t0
       
#error
mseF = torch.sum( (exact_interior - uF.forward(x_grid))**2 ) / len(x_grid)
b_errorF = torch.sum( uF.forward( bc_grid )**2 ) / len(bc_grid)

mseF = mseF.detach().numpy()
b_errorF = b_errorF.detach().numpy()
        
#converged solution
sol = uF.forward(plot_grid).detach().numpy()
ZF = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZF = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZF[i,j] = ZF[i,j]
        else:
            maskedZF[i,j] = np.nan
    
#l2 error over domain
l2_errorF = ( maskedZF-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorF , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uF.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%

######################################################

#%% TRAIN G -- AUGMENTED LAGRANGIAN

#initialize network
uG = SolutionG(n)

#initial penalty parameter
penG = 1

#training parameters
epochsG = 3000
lrG = 1e-4
tolG = 1e-6
conG = tolG + 1
lamG = torch.rand((pts-1,1))
lossesG = []

#ADAM optimization
optimizer_sol = torch.optim.Adam( uG.parameters() , lr=lrG )

count = 0

t0 = time.time()
print('begin training G...')
while conG > tolG:
    
    for i in range(epochsG):
        
        optimizer_sol.zero_grad()
        
        bulk = torch.sum( dx1**2 * uG.energy_density( x_grid ) )
        penalty = 0.5 * penG * torch.sum( torch.mul( uG.forward( bc_grid )**2 , ds ) )
        lagrange = torch.sum( torch.mul( lamG * uG.forward(bc_grid) , ds ) )
       
        loss = bulk + lagrange + penalty
        
        loss.backward()
        
        optimizer_sol.step()  
        
        lossesG += [ round( loss.item() , 4 ) ]
    
        if count % 1000 == 0:
            print(f'Epoch {i}, Loss {lossesG[count]}')  
            
        count += 1
            
    conG = float( torch.sum( uG.forward( bc_grid )**2 ) / len(bc_grid) )
    lamG = lamG + penG * uG.forward(bc_grid).detach()    
    penG = 2*penG
    
t1 = time.time()

#run time
tG = t1-t0
  
#error
mseG = torch.sum( (exact_interior - uG.forward(x_grid))**2 ) / len(x_grid)
b_errorG = torch.sum( uG.forward( bc_grid )**2 ) / len(bc_grid)

mseG = mseG.detach().numpy()
b_errorG = b_errorG.detach().numpy()
        
#converged solution
sol = uG.forward(plot_grid).detach().numpy()
ZG = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZG = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZG[i,j] = ZG[i,j]
        else:
            maskedZG[i,j] = np.nan
    
#l2 error over domain
l2_errorG = ( maskedZG-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorG , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uG.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%

######################################################

#%% TRAIN H -- NEURAL NETWORK LAGRANGE MULTIPLIER

#initialize network
uH = SolutionH(n)
lamH = LagrangeH(n)

#training parameters
epochsH = 80000
lrH = 1e-5
lossesH = np.zeros(epochsH)

#ADAM optimization
optimizer_sol = torch.optim.Adam( uH.parameters() , lr=lrH )
optimizer_lam = torch.optim.Adam( lamH.parameters() , lr=1*lrH , maximize=True )

t0 = time.time()
print('begin training H...')
for i in range(epochsH):
    
    optimizer_sol.zero_grad()
    optimizer_lam.zero_grad()
    
    bulk = torch.sum( dx1**2 * uH.energy_density( x_grid ) )
    lagrange = torch.sum( torch.mul( lamH.forward(x1_grid) * uH.forward(bc_grid) , ds ) )
   
    loss = bulk + lagrange
    
    loss.backward()
    
    optimizer_sol.step() 
    optimizer_lam.step()
    
    lossesH[i] = round( loss.item() , 4 )
    
    if i % 1000 == 0:
        print(f'Epoch {i}, Loss {lossesH[i]}')  

t1 = time.time()

#run time
tH = t1-t0
    
#error
mseH = torch.sum( (exact_interior - uH.forward(x_grid))**2 ) / len(x_grid)
b_errorH = torch.sum( uH.forward( bc_grid )**2 ) / len(bc_grid)

mseH = mseH.detach().numpy()
b_errorH = b_errorH.detach().numpy()
        
#converged solution
sol = uH.forward(plot_grid).detach().numpy()
ZH = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZH = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZH[i,j] = ZH[i,j]
        else:
            maskedZH[i,j] = np.nan
    
#l2 error over domain
l2_errorH = ( maskedZH-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorH , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uH.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%

######################################################

#%% TRAIN I -- SQP CONSTRAINED OPTIMIZATION

#initialize network
uI = SolutionI(n)

#energy objective
def obj():
    return torch.sum( dx1**2 * uI.energy_density(x_grid) )

#zero temperature boundary (constraint)
def con():
    return uI.forward(bc_grid)

#parameters for SQP optimization
scipy_minimizer_args = {
    'maxiter': 50000,
    'xtol': 1e-15,
    # 'initial_tr_radius': 0.1,
    # 'initial_constr_penalty': 0.01,
    # 'initial_barrier_parameter': 100.0,
}

#constraint violation tolerance
eps = 0.001

#create a tracker for optimization
tracker = ConstrainedOptimizerTracker( scipy_minimizer_args['maxiter'] , pts-1 , obj , con , lb=-eps , ub=eps )

#pass in the tracker's callback as the 3rd argument
optimizer = TorchScipyOptimizer( uI.parameters(), scipy_minimizer_args, tracker.callback)


#run the optimization
t0 = time.time()
optimizer.step( tracker.obj_fn , tracker.con_fn , lower_bnd=-eps , upper_bnd=eps )
t1 = time.time()

#run time
tI = t1-t0

#extract number of steps and convergence history
epochsI = tracker.count
lossesI = tracker.obj_history[:epochsI-1]

# #initialize optimizer
# slsqp_optim = TorchScipyOptimizer( uI.parameters() , scipy_minimizer_args )

# #run optimization, this variable stores summary data
# z = slsqp_optim.step( obj , con, lower_bnd=-eps , upper_bnd=eps )

#error
mseI = torch.sum( (exact_interior - uI.forward(x_grid))**2 ) / len(x_grid)
b_errorI = torch.sum( uI.forward( bc_grid )**2 ) / len(bc_grid)

mseI = mseI.detach().numpy()
b_errorI = b_errorI.detach().numpy()
        
#converged solution
sol = uI.forward(plot_grid).detach().numpy()
ZI = np.reshape( sol , (scale*pts,scale*pts) )

#for plotting only over semicircle
maskedZI = np.zeros((scale*pts,scale*pts))
for i in range(scale*pts):
    for j in range(scale*pts):
        if X2[i,j] <= semicircle(X1[i,j]):
            maskedZI[i,j] = ZI[i,j]
        else:
            maskedZI[i,j] = np.nan
    
#l2 error over domain
l2_errorI = ( maskedZI-maskedZex )**2

#plot error in interior and along boundary
fig , axs = plt.subplots( 2 , 1 )

ax1 = axs[0]
ax2 = axs[1]

cont = ax1.contourf( X1 , X2 , l2_errorI , 30 )
ax1.plot(xx,semicircle(xx),linewidth=3.5,color='k')
ax1.plot(xx,np.zeros(xx.shape),linewidth=3.5,color='k') 
ax1.axis('off') 
ax1.set_xlim(-1.05,1.05)
ax1.set_ylim(0,1.05)
ax1.set_title('L2 Error')
cbar = fig.colorbar( cont , ax=ax1 )
tick_locator = ticker.MaxNLocator(nbins=3)
cbar.locator = tick_locator
cbar.update_ticks()

ax2.plot( x1_grid.numpy() , uI.forward(bc_grid).detach().numpy() )
ax2.set_xlabel('s')
ax2.set_ylabel('u(s)')
ax2.set_title('Temperature Along Boundary')
ax2.set_xticks([])

plt.show()

#%%

######################################################

#%% PLOTTING

plt.close('all')

#ignore early loss values that are large
thresh = 2
lossesA = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesA ] )
lossesB = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesB ] )
lossesC = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesC ] )
lossesD = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesD ] )
lossesE = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesE ] )
lossesF = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesF ] )
lossesG = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesG ] )
lossesH = np.asarray( [ np.nan if elem > thresh else elem for elem in lossesH ] )

#training
horizon = 30000
plt.figure()
plt.plot( lossesA[:horizon] , color='r' , label='Hard Enforcement' )
plt.plot( lossesB[:horizon] , color='b' , label='Penalty' )
plt.plot( lossesC[:horizon] , color='k' , label='SA PINN' )
plt.plot( lossesD[:horizon] , color='m' , label='Discrete Lagrange' )
plt.plot( lossesE[:horizon] , color='c' , label='Shape Function Lagrange' )
plt.plot( lossesF[:horizon] , color='g' , label='Nitsche Method' )
plt.plot( lossesG[:horizon] , color='y' , label='Augmented Lagrangian' )
plt.plot( lossesH[:horizon] , color='purple' , label='Neural Network Lagrange' )
plt.plot( energy*np.ones(horizon) , '--' , label='Exact energy' )
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Energy')
plt.title('Training Convergence')
plt.show()


#comparison of methods
sz = 150
plt.figure()
plt.scatter( tA , mseA , s=sz , label='Hard enforcement' )
plt.scatter( tB , mseB , s=sz , label='Penalty' )
plt.scatter( tC , mseC , s=sz , label='SA PINN' )
plt.scatter( tD , mseD , s=sz , label='Discrete Lagrange' )
plt.scatter( tE , mseE , s=sz , label='Shape Function Lagrange' )
plt.scatter( tF , mseF , s=sz , label='Nitsche Method' )
plt.scatter( tG , mseG ,  s=sz , label='Augmented Lagrangian' )
plt.scatter( tH , mseH , s=sz , label='Neural Network Lagrange' )
plt.scatter( tI , mseI , s=sz , label='Constrained Optimization' )
plt.yscale('log')
plt.legend()
plt.xlabel('Run Time (s)')
plt.ylabel('MSE')
plt.title('Comparison of Solution Errors')
plt.show()

plt.figure()
plt.scatter( tA , b_errorA , s=sz , label='Hard enforcement' )
plt.scatter( tB , b_errorB , s=sz , label='Penalty' )
plt.scatter( tC , b_errorC , s=sz , label='SA PINN' )
plt.scatter( tD , b_errorD , s=sz , label='Discrete Lagrange' )
plt.scatter( tE , b_errorE , s=sz , label='Shape Function Lagrange' )
plt.scatter( tF , b_errorF , s=sz , label='Nitsche Method' )
plt.scatter( tG , b_errorG , s=sz , label='Augmented Lagrangian' )
plt.scatter( tH , b_errorH , s=sz , label='Neural Network Lagrange' )
plt.scatter( tI , b_errorI , s=sz , label='Constrained Optimization' )
plt.yscale('log')
plt.legend()
plt.xlabel('Run Time (s)')
plt.ylabel('Dirichlet Boundary MSE')
plt.title('Comparison of Boundary Condition Errors')
plt.show()

#%%

######################################################












