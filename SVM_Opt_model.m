classdef svmOpt
    properties
        H; % H_ij = y_i*y_j*x_i*x*j
        X; %data -features
        y; %label
        size; %size of dataset
        C; % free parameter which controls the trade-off between the slack 
        %variable penalty and the size of the margin
        A; % variables for determining support vectors
        L; %loss function
        SVs; %support vector sets
        isSV;%binary flagto check if it is support vector
        b; %bias terms at decision boundary
        kernel;
        numclass;
        outlierFraction; %outlier fraction of one-class svm.
        optimizer; % Trust-region style Quasi-Newton methods(SR1)
        %/Conjugate Gradient Methods
        e; %elaspe time of training
    end
    
    methods
        function obj = svmOpt(X, y, kernelName, C, outlierFraction, optimizer)
            obj.X = X;
            obj.y = y;
            size_X = size(obj.X);
            obj.size = size_X(1);
            obj.numclass = length(unique(y));
            
            switch kernelName
                case 'linear'
                    obj.kernel = @(x1, x2) x1 * (x2.');
                case 'RBF'
                    sigma = 1;
                    obj.kernel = @(x1, x2) exp(-norm(x1-x2)/(2*sigma));
            end
            
            H = setH(obj);
            obj.H = H;
            
            obj.C = C;
            obj.outlierFraction = outlierFraction;
            obj.optimizer = optimizer;
            
            if obj.outlierFraction >0
                obj.C = 1/(length(obj.y)*obj.outlierFraction);
            end
                                    
            obj.A = zeros(obj.size, 1)-1;
           
            obj.L = QudraticPenalty(obj);
            
            [obj.A, obj.e] = fit(obj);
            [obj.SVs, obj.isSV] = assignSupportVecotor(obj);
            obj.b = setBias(obj);
        end
        
        function [ aMin, e] = fit(obj)
            t = cputime;           
            mu0 = 1;
            a0 = obj.A;
            [aMin, fMin_QP, nIter_QP, info_QP] = PenaltyAugmented(obj.L, mu0, a0, 'QuadraticPenalty', obj.optimizer);                       
            e = cputime-t;
            disp('finish training')
% Interior method: wait for fixing, buggy now.
%             lambda0 = zeros(2*length(obj.A), 1) + 10;
%             lambda0 = [-1/(-obj.A) -1/(obj.A-obj.C) ].';
%             mu = 10;
%             tol = 1e-8;
%             tolFeas = 1e-6;
%             maxIter = 100;
%             opts.maxIter = 100;
%             opts.alpha = 0.01;
%             opts.beta = 0.5;
%             [aMin, fMin, t, nIter, infoPD] = interiorPoint_PrimalDual(obj.L.F, obj.L.ineqConstraint, obj.L.eqConstraint, obj.A, lambda0, nu0, mu, tol, tolFeas, maxIter, opts)
            
            
        end
        
        function logits = predict(obj, X)
            logits = [];            
            if obj.outlierFraction == 0                 
                for s=1:length(X)
                    tot_sum = 0;
                    for m=1:obj.size
                       tot_sum = tot_sum + obj.A(m) .* obj.y(m) .* obj.kernel(obj.X(m,:), X(s,:));
                    end
                    logits = [logits, tot_sum+obj.b];
                end
            elseif obj.outlierFraction > 0
                for s=1:length(X)
                    tot_sum = 0;
                    for m=1:obj.size
                       tot_sum = tot_sum + obj.A(m) .* obj.kernel(obj.X(m,:), X(s,:));
                    end
                    logits = [logits, tot_sum-obj.b];
                end
            end
        end
        
        
        function L = QudraticPenalty(obj)
            
            if obj.outlierFraction == 0           
                L.f = @(a, mu) 0.5*sum((a * a.').*obj.H, 'all') - sum(a, 'all') ...
                    + (mu/2)*((a.' * obj.y).^2 + sum(max(0, 0-a).^2, 'all') ...
                    + sum(max(0, a-obj.C).^2, 'all') );

                str = '@(a, mu, H, y, C)[';
                for i=1:obj.size
                    df_i = sprintf("0.5*(sum(a.'.*H(%d,:), 'all') + a.'*H(:,%d)) - 1 + mu*((a.' * y)*y(%d)+min(0, a(%d))+max(0, a(%d)-C));",i,i,i,i,i);
                    str = strcat(str,df_i);
                end
                str = strcat(str,"];");
                df = str2func(str);
                L.df = @(a, mu) df(a, mu, obj.H, obj.y, obj.C);
            elseif obj.outlierFraction > 0
                L.f = @(a, mu) 0.5*sum((a * a.').*obj.H, 'all') ...
                    + (mu/2)*((sum(a,'all')-1).^2 + sum(max(0, 0-a).^2, 'all') ...
                    + sum(max(0, a-obj.C).^2, 'all') );
                
                str = '@(a, mu, H, C)[';
                for i=1:obj.size
                    df_i = sprintf("0.5*(sum(a.'.*H(%d,:), 'all') + a.'*H(:,%d)) + mu*((sum(a,'all')-1)+min(0, a(%d))+max(0, a(%d)-C));",i,i,i,i);
                    str = strcat(str,df_i);
                end
                str = strcat(str,"];");
                df = str2func(str);
                L.df = @(a, mu) df(a, mu, obj.H, obj.C);
            end
        end
        
        function L = InteriorPoint_PrimalDual(obj)
            L.F.f = @(a) 0.5*sum((a * a.').*obj.H, 'all') - sum(a, 'all');
            str = '@(a, H)[';
            for i=1:obj.size
                df_i = sprintf("sum(a.'.*H(%d,:), 'all') - 1;",i);
                str = strcat(str,df_i);
            end
            str = strcat(str,"];");
            df = str2func(str);
            L.F.df = @(a) df(a, obj.H);
            L.F.d2f = @(a) obj.H;
            
            ineqConstraint.f = @(a, C) [a.', a.'-C].';
            ineqConstraint.df = @(a) [diag(zeros(1, length(a))-1);  diag(zeros(1, length(a))+1)];
            ineqConstraint.d2f = @(a) zeros(length(obj.A),length(obj.A), 2*length(obj.A));
            
            L.ineqConstraint.f = @(a) ineqConstraint.f(a, obj.C);
            L.ineqConstraint.df = ineqConstraint.df;
            L.ineqConstraint.d2f = ineqConstraint.d2f;
            L.eqConstraint.A = obj.y.';
            L.eqConstraint.b = zeros(1, 1);
        end
                
        function H = setH(obj)            
            H = zeros(obj.size, obj.size);
            
            for i=1:obj.size
                for j=1:obj.size                              
                    H(i,j) = obj.y(i)* obj.y(j)* obj.kernel(obj.X(i,:), obj.X(j,:));  
                end
            end            
        end
               
        function [SVs, isSV] = assignSupportVecotor(obj)
            SVs = [];
            isSV = zeros(obj.size, 1);
            for i=1:obj.size
                if obj.C > 0
                    if obj.A(i) < obj.C && obj.A(i) > 0
                        SVs = [SVs; obj.X(i, :)];
                        isSV(i) = 1;
                    end
                elseif obj.C == 0
                    if obj.A(i) > 0
                       SVs = [SVs obj.X(i, :)];
                       isSV(i) = 1;
                    end
                end
            end                                        
        end 
        
        function b = setBias(obj)               
            temp = [];
            y_s = obj.isSV .* obj.y; % 100x1            
            x_s = (obj.isSV .* obj.X); %2x100
            if obj.outlierFraction == 0                   
                temp = [];
                for s=1:obj.size
                    tot_sum = 0;
                    if x_s(s,1) == 0 && x_s(s,2) == 0
                        temp = [temp 0];
                       continue
                    end
                    for m=1:obj.size
                       tot_sum = tot_sum + obj.A(m) .* obj.y(m) .* obj.kernel(obj.X(m,:), x_s(s,:));
                    end
                    temp = [temp tot_sum];
                end
                b = sum(y_s - temp.')/sum(obj.isSV);   
            elseif obj.outlierFraction > 0 
                for s=1:obj.size
                    tot_sum = 0;
                    if x_s(s,1) == 0 && x_s(s,2) == 0
                        temp = [temp 0];
                        continue
                    end
                    for m=1:obj.size

                       tot_sum = tot_sum + y_s(s).*obj.A(m) .* obj.kernel(obj.X(m,:), x_s(s,:));
                    end                    
                    temp = [temp tot_sum];
                    
                end
                b = sum(temp)/sum(obj.isSV);                                
            end
        end
        
    end
end
