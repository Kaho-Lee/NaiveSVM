function [xMin, fMin, nIter, info] = PenaltyAugmented(Q, mu0,x0, type, varargin)
    % Initialisation
    alpha0 = 1;
    maxIter = 100;
    alpha_max = alpha0;
    tol = 1e-6;
    
    tau = 1e-6;
    epsilon = 1e-10;
  
    if strcmp(type, 'AugmentedLagrangian')
        v_k = varargin{1};
        c_1 = varargin{2};
        info.vks = v_k;
    end
    
    
    % Steepest descent line search strong WC
    lsOptsSteep.c1 = 1e-4;
    lsOptsSteep.c2 = 0.9;
    
    % Initialization
    nIter = 0;
    x_k = x0;
    mu_k = mu0;
    info.xs = x0;
    info.mus = mu0;
    stopCond = false; 
    
    
    while(~stopCond && nIter <= maxIter)
        nIter = nIter + 1;
        switch type
            case 'QuadraticPenalty'
                cur_Q.f = @(x) Q.f(x, mu_k);
                cur_Q.df = @(x) Q.df(x, mu_k);
                %cur_Q.d2f = @(x) Q.d2f(x, mu_k);
                %lsFun = @(x_k, p_k, alpha0) lineSearch(cur_Q, x_k, p_k, alpha_max, lsOptsSteep);
                %[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(cur_Q, 'bfgs', lsFun, alpha0, x_k, tol, maxIter, tau);
                
                % Trust region parameters 
                eta = 0.1;  % Step acceptance relative progress threshold
                Delta = 1; % Trust region radius
                debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

                % Minimisation with 2d subspace and dogleg trust region methods
                Fsr1 = cur_Q;
                [xSteep, fSteep, nIterSteep, infoSteep] = trustRegion(Fsr1, x_k, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, tau, debug);
                
                mu_k = mu_k*1.5;
            case 'AugmentedLagrangian'
                cur_Q.f = @(x) Q.f(x, mu_k, v_k);
                cur_Q.df = @(x) Q.df(x, mu_k, v_k);
                %cur_Q.d2f = @(x) Q.d2f(x, mu_k, v_k);
                %lsFun = @(x_k, p_k, alpha0) lineSearch(cur_Q, x_k, p_k, alpha_max, lsOptsSteep);
                %[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(cur_Q, 'bfgs', lsFun, alpha0, x_k, tol, maxIter, tau);
                
                eta = 0.1;  % Step acceptance relative progress threshold
                Delta = 1; % Trust region radius
                debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

                % Minimisation with 2d subspace and dogleg trust region methods
                
                Fsr1 = cur_Q;
                [xSteep, fSteep, nIterSteep, infoSteep] = trustRegion(Fsr1, x_k, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, tau, debug);
                
                tau = norm(c_1(x_k));
                v_k = v_k - mu_k*c_1(xSteep);
                mu_k = mu_k*1.5;
                info.vks = [info.vks v_k];
%                 disp('haha')
%                 disp(c_1(x_k), tau)
        end
        
        if norm(xSteep-x_k, 2) <= epsilon
            stopCond = true; 
        end
        

        
        x_k = xSteep;
        info.mus = [info.mus, mu_k];
        info.xs = [info.xs, x_k];
    end
    
    xMin = xSteep;
    fMin = fSteep;
    q = norm(cur_Q.df(xSteep), 2);
    
end