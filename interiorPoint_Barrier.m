function [xMin, fMin, t, nIter, infoBarrier] = interiorPoint_Barrier(F, phi, x0, t, mu, tol, maxIter)
% INTERIORPOINT_BARRIER function to minimise a quadratic form with constraints
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% phi: structure with fields
%  - f: barrier function
%  - df: gradient of barrier function
%  - d2f: Hessian of barrier function
% x0: initial iterate
% t: parameter for the barrier
% mu: increase factor for t
% tol: tolerance on the (duality gap ~ m/t) scaled with 1/m, m #inequality constaints. 
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoBarrier: structure with information about the iteration 
%   - xs: iterate history for x 
%   - ys: iterate history for y
%
% Copyright (C) 2017  Kiko Rul·lan, Marta M. Betcke

% Initilize
nIter = 0;
stopCond = false;
x_k = x0;
infoBarrier.xs = x_k;
infoBarrier.fs = F.f(x_k);
infoBarrier.inIter = 0;
infoBarrier.dGap = 1/t;

% Parameters for centering step
alpha0 = 1; 
opts.c1 = 1e-4;
opts.c2 = 0.9;
opts.rho = 0.5;
tolNewton = 1e-12;
maxIterNewton = 100;
% Loop 
while (~stopCond && nIter < maxIter)
    disp(strcat('Iteration ', int2str(nIter)));
    % Create function handler for centering step (needs to be redifined at each step because of changing "t")
    G.f = @(x) t*F.f(x) + phi.f(x);
    G.df = @(x) t*F.df(x) + phi.df(x);
    G.d2f = @(x) t*F.d2f(x) + phi.d2f(x);
    
    % Line search function (needs to be redefined at each step because of changing G) 
    %lsFun = @(x_k, p_k, alpha0) lineSearch(G, x_k, p_k, alpha0, opts);
    lsFun = @(x_k, p_k, alpha0) backtracking(G, x_k, p_k, alpha0, opts);
    % Centering step
    [x_k, f_k, nIterLS, infoIter] = descentLineSearch(G, 'newton', lsFun, alpha0, x_k, tolNewton, maxIterNewton);   
    
    % Check stopping condition (m/t). Assumes the tolerance has been scaled with 1/m
    if 1/t < tol; stopCond = true; end
   
    % Increase t
    t = mu*t;
    
    % Store info
    infoBarrier.xs = [infoBarrier.xs x_k];
    infoBarrier.fs = [infoBarrier.fs f_k];
    infoBarrier.inIter = [infoBarrier.inIter nIterLS];
    infoBarrier.dGap = [infoBarrier.dGap 1/t];
    
    % Increment number of iterations
    nIter = nIter + 1;
end

% Assign values
xMin = x_k;
fMin = F.f(x_k);

