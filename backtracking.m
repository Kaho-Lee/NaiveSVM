function [alpha, info] = backtracking(F, x_k, p, alpha0, opts)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
% x_k: current iterate
% p: descent direction
% alpha0: initial step length 
% opts: backtracking specific option structure with fields
%   - rho: in (0,1) backtraking step length reduction factor
%   - c1: constant in sufficient decrease condition f(x_k + alpha_k*p) > f(x_k) + c1*alpha_k*(df_k'*p)
%         Typically chosen small, (default 1e-4). 
%
% OUTPUTS
% alpha: step length
% info: structure with information about the backtracking iteration 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan


% ====================== YOUR CODE HERE ======================
    alpha_k = alpha0;
    info.alphas = [];
    info.alphas = [info.alphas, alpha_k];
    df_k = F.df(x_k);
    while (min(x_k + alpha_k.*p) < 0) || (max(x_k + alpha_k.*p) > F.C)
        alpha_k = alpha_k * opts.rho;
    end
    while F.f(x_k + alpha_k.*p) > F.f(x_k) + opts.c1*alpha_k*(df_k'*p)
        alpha_k = alpha_k * opts.rho;
        info.alphas = [info.alphas, alpha_k];
    end
    alpha = alpha_k;
% ============================================================

end