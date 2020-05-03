function [x, Fx, mse] = gradientDescent(b, F, lambda, x0, stop, tol, maxit, xt)
% GRADIENTDESCENT Gradient descent algorithm for minimizing F(x)
%  [x, Fx, err] = gradientDescent(b, F, lambda, x0, stop, tol, maxit, xt)
%  Solves
%   arg min_x F(x,lambda,b)  over all x in R^n
%  with Gradient Descent.
%
%  INPUTS:
%   b      - data vector
%   F      - structure containing information about the function F:
%             F.f  - function
%             F.df - gradient
%             F.L  - Lipschitz constant of the gradient
%   lambda - regularization parameter
%   x0     - initial solution (usually 0)
%   stop   - stopping condition, choose from
%             0 - relative variation in number of nnz components falls below tol
%             1 - relative variation in regularized objective function falls below tol
%             2 - relative variation in estimate falls below tol (default)
%             3 - value of the regularized objective function  falls below tol             
%   tol    - stopping tolerance for criterium specified with stop
%   maxit  - maximal number of ISTA iterations
%   xt     - true solution, if available the error will be computed
%  OUTPUTS:
%   x      - solution x 
%   Qx     - values of Q(x_j) over the iteration
%   mse    - mean square error if true solution xt specified, otherwise ||iPsi(x)||^2/length(xt)
%
% Copyright (C) 2013 Marta M. Betcke

% Default values of parameters
DL = 1;        % Lipschitz constant
DTOL = 1e-4;   % Tolerance for the objective function change, STOP if below
DMAXIT = 200;  % Maximal number of iterations 
DSTOP = 1;     % Stopping condition, relative variation in the estimate

% If solution not known, return ||x||_2 in err
if nargin < 8 
  xt = zeros(n,1);
end
% Assign default values of the parameters
if nargin < 7 
  maxit = DMAXIT;
end
if nargin < 6 
  tol = DTOL;
end
if nargin < 5
  stop = DSTOP;
end
if ~isfield(F,'L')
  F.L = DL;
end

% Initialization
mse = zeros(1,maxit);
Fx = zeros(1,maxit);
x = x0;

% Evaluate objective function
Fx(1) = F.f(x,lambda,b);
% Error
mse(1) = sum((x-xt).^2);

% Maximal step size 1/L, L = lambda_max(A^T A) Lipschitz constant of the gradient
t = 1/F.L;

for j = 2:maxit
  % Gradient step
  xnew = x - t*F.df(x, lambda, b);
  % Evaluate objective function
  Fx(j) = F.f(xnew, lambda, b);
  % Error if solution xt known
  mse(j) = sum((xnew-xt).^2);

  %Stopping criterium
  switch stop
    case 0,
      % Relative variation of the number of nnz components
      criterion = sum(nnz(xnew) ~= nnz(x)) / nnz(x);
    case 1,
      % Relative variation of the regularized objective function
      criterion = abs(Fx(j-1) - Fx(j))/Fx(j-1);
    case 2,
      % Relative variation of the estimate
      criterion = norm(xnew - x)/norm(x);
    case 3,
      % Value of the objective function
      criterion = Fx(j);
    otherwise,
      error(['Invalid stopping criterion']);
  end

  % Reassign
  x = xnew;

  if criterion < tol
    break
  end    

%  % Stopping criterium: objective function change smaller than tol
%  if abs(Fx(j-1) - Fx(j)) < tol
%    break
%  end
end

% Remove tailing 0s
mse = mse(1:j);
Fx = Fx(1:j);

% MSE
mse = mse/length(xt);