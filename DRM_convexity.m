clear; close all;

%integration grid
pts = 50;
grid = linspace(0,1,pts);
dx = grid(2)-grid(1);
grid = grid + dx/2;
grid = grid(1:end-1);

%discretization degrees of freedom and spatial coordinate
N = 4;
a = sym( 'a' , [ N , 1 ] ); 
x = sym('x');

%define crazy nonlinear discretization here
u = ( a(1)*x.^2 + sin(a(2))*x/a(3) + tanh(a(4)*x) ) .* sin(pi*x);

% %fourier discretization to verify code (uncomment to try it)
% u = ( a(1)*sin(pi*x) + a(2)*sin(2*pi*x) + a(3)*sin(3*pi*x) + a(4)*sin(4*pi*x) );

%compute symbolic derivatives
u_x = diff( u , x , 1 );
u_xx = diff( u , x , 2 );
u_xa = jacobian( u_x , a )';
u_a = jacobian( u , a )';
u_aa = jacobian( u_a , a );

%convert symbolic expressions to numerical functions
u = matlabFunction( u , 'vars' , { a , x } );
u_x = matlabFunction( u_x , 'vars' , { a , x } );
u_xx = matlabFunction( u_xx , 'vars' , { a , x } );
u_xa = matlabFunction( u_xa , 'vars' , { a , x } );
u_a = matlabFunction( u_a , 'vars' , { a , x } );
u_aa = matlabFunction( u_aa , 'vars' , { a , x } );

%energy as a function of degrees of freedom (constant forcing)
energy = @(a) ( -dx*sum( u(a,grid) ) + 0.5*dx*sum( u_x(a,grid).^2 ) );

%norm of energy gradient
kkt = @(a) ( norm( dx * u_xa(a,grid) * u_x(a,grid)' - dx * sum( u_a(a,grid) , 2 )  ) );

%initial guess of parameteres
a0 = randn(N,1);

%optimization options
evals = 5E4;
options = optimoptions('fmincon', ...
    'OptimalityTolerance', 1e-9, ...
    'GradObj' , 'off' , ...
    'StepTolerance', 1e-9, ...
    'MaxFunctionEvaluations', evals,...
    'MaxIterations', evals);

%find parameters that minimize energy
A = fmincon( kkt , a0 , [] , [] , [] , [] , [] , [] , [] , options );

%plot solution
figure()
plot( grid , u(A,grid) )
hold on
plot( grid , -0.5*grid.^2 + 0.5*grid )
legend('computed','exact')
xlabel('x')
ylabel('u(x)')
title('Solution')

%compute positive definite and weird almost weak form matrices
PD = zeros( length(A) , length(A) );
R = zeros( length(A) , length(A) );

%integrate to form second derivative matrices at solution parameters A
for i=1:(pts-1)
    R = R + dx * ( ( u_xx(A,grid(i)) + 1 ) * u_aa(A,grid(i)) );
    PD = PD + dx * ( u_xa(A,grid(i)) * u_xa(A,grid(i))' );
end

%hessian matrix
H = PD - R;

%eigenvalues of hessian
%when eigenvalues are all positive, the solution is a minimum
%when eigenvalues are all negative, the solution is a maximum
%when eigenvalues are both positive and negative, the solution is a saddle
eig(H)

%value of energy at solution
%larger energy values indicate a worse solution
energy(A)



