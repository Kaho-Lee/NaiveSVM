function [xMin, fMin, nIter, info] = nonlinearConjugateGradient(F, ls, type, alpha0, x0, tol, maxIter, tau)
% NONLINEARCONJUGATEGRADIENT Wrapper function executing conjugate gradient with Fletcher Reeves algorithm
% [xMin, fMin, nIter, info] = nonlinearConjugateGradient(F, ls, 'type', alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% ls: handle to linear search function
% type: beta update type {'FR', 'PR'}
% alpha0: initial step length 
% rho: in (0,1) backtraking step length reduction factor
% c1: constant in sufficient decrease condition f(x_k + alpha_k*p_k) > f_k + c1*alpha_k*(df_k')*p_k)
%     Typically chosen small, (default 1e-4). 
% x0: initial iterate
% tol: stopping condition on relative error norm tolerance 
%      norm(x_Prev - x_k)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Kiko Rullan, Marta M. Betcke 

% Parameters
% Stopping condition {'step', 'grad'}
stopType = 'grad';

stopType = 'PenaltyAugmented';

decay = 0.9;

% Initialization
nIter = 0;
normError = 1;
x_k = x0;
df_k = F.df(x_k);
p_k = -df_k;
info.xs = x0;
info.alphas = alpha0;
info.betas = [];
info.cos_theta = [(-df_k'*p_k)./(norm(df_k)*norm(p_k))];
    
info.pk = p_k;
stopCond = 0;

% Loop until convergence or maximum number of iterations
while (~stopCond && nIter <= maxIter)

    %============================ YOUR CODE HERE =========================================
    switch type
        case 'PR'
            nIter = nIter + 1;
            alpha_k = ls(x_k, p_k, alpha0);
            x_k_1 = x_k;
            x_k = x_k + alpha_k*p_k;
            df_k1 = F.df(x_k);
            beta_k1_PR = (df_k1'*(df_k1 - df_k))/norm(df_k)^2;
            beta_k1_FR = (df_k1'*df_k1)/(df_k'*df_k);
            %beta_k1 = beta_k1_PR;
            
            cos_theta = abs(df_k'*df_k1)/norm(df_k)^2;
            if cos_theta > 0.1
                beta_k1 = 0;
            else
                beta_k1 = beta_k1_PR;
%                 if beta_k1_PR < -beta_k1_FR
%                     beta_k1 = -beta_k1_FR;
%                 elseif abs(beta_k1_PR) <= beta_k1_FR
%                     beta_k1 = beta_k1_PR;
%                 else
%                     beta_k1 = beta_k1_FR;
%                 end
                    
            end
            
            %beta_k1 = max([beta_k1_PR 0]);
            p_k = -df_k1 + beta_k1*p_k;
            df_k = df_k1;

            info.xs = [info.xs x_k];
            info.alphas = [info.alphas alpha_k];
            info.betas = [info.betas beta_k1];
            info.cos_theta = [info.cos_theta (-df_k'*p_k)./(norm(df_k)*norm(p_k))];
        case 'FR'
            nIter = nIter + 1;
            alpha_k = ls(x_k, p_k, alpha0);
            x_k_1 = x_k;
            x_k = x_k + alpha_k*p_k;
            df_k1 = F.df(x_k);
            beta_k1 = (df_k1'*df_k1)/(df_k'*df_k);
            p_k = -df_k1 + beta_k1*p_k;
            df_k = df_k1;

            info.xs = [info.xs x_k];
            info.alphas = [info.alphas alpha_k];
            info.betas = [info.betas beta_k1];
            info.cos_theta = [info.cos_theta (-df_k'*p_k)./(norm(df_k)*norm(p_k))];
    end

    %=====================================================================================
    switch stopType
      case 'step' 
        % Compute relative step length
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        stopCond = (normStep < tol);
      case 'grad'
        stopCond = (norm(df_k, 'inf') < tol*(1 + abs(F.f(x_k))));
      case 'PenaltyAugmented'
        norm_qSteep = norm(F.df(x_k), 2);
        
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        
        stopCond = (norm_qSteep <= tau) | (normStep < tol);
        tau = tau * decay;
    end
end

% Assign output values 
xMin = x_k;
fMin = F.f(x_k); 