function [xMin, fMin, nIter, info] = PenaltyAugmented(Q, mu0,x0, type, optimizer, varargin)
    % Initialisation
    alpha0 = 1;
    maxIter = 100;
    alpha_max = alpha0;
    tol = 1e-6;
    
    tau = 1e-6;
    epsilon = 1e-6;
  
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
        
            
        cur_Q.f = @(x) Q.f(x, mu_k);
        cur_Q.df = @(x) Q.df(x, mu_k);
        switch optimizer
            case 'BFGS'
                % Trust region parameters 
                eta = 0.1;  % Step acceptance relative progress threshold
                Delta = 1; % Trust region radius
                debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

                % Minimisation with 2d subspace and dogleg trust region methods
                lsFun = @(x_k, p_k, alpha0) lineSearch(cur_Q, x_k, p_k, alpha_max, lsOptsSteep);
                [cur_x, cur_f, nIterSteep, infoSteep] = descentLineSearch(cur_Q, 'bfgs', lsFun, alpha0, x_k, tol, maxIter, tau);

                %Fsr1 = cur_Q;
                %[cur_x, cur_f, nIterSteep, infoSteep] = trustRegion(Fsr1, x_k, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, tau, debug);
            case 'ConjugateGrad'                        
                lsOptsCG_LS.c1 = 1e-4;
                lsOptsCG_LS.c2 = 0.1;
                lsFun = @(x_k, p_k, alpha0) lineSearch(cur_Q, x_k, p_k, alpha_max, lsOptsCG_LS);
                [cur_x, cur_f, nIterCG_FR_LS, infoCG_FR_LS] = nonlinearConjugateGradient(cur_Q, lsFun, ...
                    'FR', alpha0, x_k, tol, maxIter,tau);
        end
                                                    
        
        
        if norm(cur_x-x_k, 2) <= epsilon
            stopCond = true; 
        end
             
        x_k = cur_x;
        if mu_k < 100   
            mu_k = mu_k*1.5;
        end
        
        info.mus = [info.mus, mu_k];
        info.xs = [info.xs, x_k];
    end
    
    xMin = cur_x;
    fMin = cur_f;
    
end