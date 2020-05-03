function [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter, tau)
% DESCENTLINESEARCH Wrapper function executing  descent with line search
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% descent: specifies descent direction {'steepest', 'newton', 'bfgs'}
% ls: specifies line search algorithm
% alpha0: initial step length 
% x0: initial iterate
% tol: stopping condition on minimal allowed step
%      norm(x_k - x_k_1)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan

% Parameters
% Stopping condition {'step', 'grad'}
stopType = 'grad';
% stopType = 'newton decrement';

stopType = 'PenaltyAugmented';
decay = 0.9;

% Extract inverse Hessian approximation handler
extractH = 1;

% Initialization
nIter = 0;
x_k = x0;
info.xs = x0;
info.alphas = alpha0;
stopCond = false; 

switch lower(descent)
  case 'bfgs'
    H_k = @(y) y;
    % Store H matrix in columns
    info.H = [];
end

% Loop until convergence or maximum number of iterations
while (~stopCond && nIter <= maxIter)
    
  % Increment iterations
    nIter = nIter + 1;

    % Compute descent direction
    switch lower(descent)
      case 'steepest'
        p_k = -F.df(x_k); % steepest descent direction
      case 'newton'
        p_k = -F.d2f(x_k)\F.df(x_k); % Newton direction
        if p_k'*F.df(x_k) > 0 % force to be descent direction (only active if F.d2f(x_k) not pos.def.)
          p_k = -p_k;
        end        
      case 'newton_logbarrier'
           size_a = size(F.z);
           first_row = [F.d2f(x_k), F.z'];
           second_row = [F.z, zeros(size_a(1), size_a(1))];
           block_matrix = [first_row; second_row];
           residual_mat = -1 *[F.df(x_k); zeros(size_a(1), 1)];
           det_block = det(block_matrix)
%            disp(size(block_matrix))
%            disp(size(residual_mat))
           if det_block == 0
               inv_block = pinv(block_matrix);
               change_x_v = inv_block * residual_mat;
           else
               change_x_v  = inv(block_matrix) * residual_mat;
           end
           p_k = change_x_v(1:length(x_k));
           newton_decrement = -F.df(x_k)'*p_k;
%            size(p_k)
           
      case 'bfgs'
        %======================== YOUR CODE HERE ==========================================
        if nIter == 1
            p_k = -H_k(1)*F.df(x_k);
        else
            p_k = -H_k(H)*F.df(x_k);
        end
        %==================================================================================

    end
    
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and f_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;
    
    switch lower(descent)
      case 'bfgs'
          
        %======================== YOUR CODE HERE ==========================================
        s_k = x_k - x_k_1;
        y_k = F.df(x_k) - F.df(x_k_1);
        rho = 1/(transpose(y_k)*s_k);

        %==================================================================================
        
        if nIter == 1
          % Update initial guess H_0. Note that initial p_0 = -F.df(x_0) and x_1 = x_0 + alpha * p_0.
          disp(['Rescaling H0 with ' num2str((s_k'*y_k)/(y_k'*y_k)) ])
          H_k = @(x) (s_k'*y_k)/(y_k'*y_k) * x;
        end
        
        
        %======================== YOUR CODE HERE ==========================================
        if nIter == 1
            H = H_k(eye(length(x0)));
            if extractH
                % Extraction of H_k as handler
                info.H{length(info.H)+1} = H_k;
            end
        end
        
        H_k = @(H) (eye(length(x0)) - rho*s_k*y_k') * H * (eye(length(x0)) - rho*y_k*s_k') + rho*s_k*transpose(s_k);
        H = H_k(H);
        %==================================================================================
        
        if extractH
            % Extraction of H_k as handler
            info.H{length(info.H)+1} = H_k;
        end
    end
    

    % Store iteration info
    info.xs = [info.xs x_k];
    info.alphas = [info.alphas alpha_k];
    
    switch stopType
      case 'step' 
        % Compute relative step length
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        stopCond = (normStep < tol);
      case 'newton decrement'
        normStep = norm(x_k - x_k_1)/norm(x_k_1);
        stopCond = (normStep < 1e-6) || (newton_decrement < tol)
      case 'grad'
        stopCond = (norm(F.df(x_k), 'inf') < tol*(1 + abs(F.f(x_k))));
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
% info.H_value = H;
end