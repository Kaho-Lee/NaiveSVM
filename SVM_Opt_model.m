classdef SVM_Opt_model
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
        optimizer; % Quasi-Newton methods(BFGS)
        %/Conjugate Gradient Methods
        e; %elaspe time of training
        fMin;
        nIter;
        info;
        QP_qCon; %convergence data placeholder, ignore the name
        QP_grad; %convergence data placeholder, ignore the name
        KernelMatrix; % K_ij = <x_i, x_j>
        method;
    end
    
    methods
        function obj = SVM_Opt_model(X, y, kernelName, C, outlierFraction, method, optimizer)
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
            
            [obj.H, obj.KernelMatrix] = setH(obj);
            
            
            obj.C = C;
            obj.outlierFraction = outlierFraction;
            obj.optimizer = optimizer;
            obj.method = method;
            
            
            if obj.outlierFraction >0
                obj.C = 1/(length(obj.y)*obj.outlierFraction);
            end
                                    
            obj.A = zeros(obj.size, 1);
           
            obj.L = QudraticPenalty(obj);
            
            if strcmp(obj.method, 'QuadraticPenalty')           
                [obj.A, obj.fMin, obj.nIter, obj.info, obj.e] = fit(obj);
                [obj.SVs, obj.isSV] = assignSupportVecotor(obj);
                obj.b = setBias(obj);
                [obj.QP_qCon, obj.QP_grad] = ConvergenceAnalysis(obj);
            
            elseif strcmp(obj.method, 'SMO')
                [obj.A, obj.b, obj.info, obj.e] = fit_SMO(obj);
                [obj.SVs, obj.isSV] = assignSupportVecotor(obj);
                [obj.QP_qCon, obj.QP_grad] = ConvergenceAnalysis(obj);
            end
        end
        
        function [A, b, info, e] = fit_SMO(obj)
            t = cputime;
            tol = 0.001;
            max_passes = 100;
            [A, b, info] = SMO( obj.y, obj.C, tol, max_passes, obj.size, obj.KernelMatrix);
            e = cputime-t;
        end
        
        function [ aMin,fMin_QP, nIter_QP, info_QP, e] = fit(obj)
            t = cputime;
            
            mu0 = 1;
            a0 = obj.A;
            [aMin, fMin_QP, nIter_QP, info_QP] = PenaltyAugmented(obj.L, mu0, a0, 'QuadraticPenalty', obj.optimizer);                       
            
            e = cputime-t;
            disp('finish training')
            
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
        
        function [QP_qCon, QP_grad] = ConvergenceAnalysis(obj)
            QP_qCon = [];
            QP_grad = [];
            if strcmp(obj.method, 'QuadraticPenalty')
                for i=2:length(obj.info.xs(1,:))
                    QP_qCon = [QP_qCon norm(obj.info.xs(:, i)-obj.A)/norm(obj.info.xs(:, i-1)-obj.A)];
                end
                for i = 1: length(obj.info.mus)
                    QP_grad = [QP_grad norm(obj.L.df(obj.info.xs(:,i), obj.info.mus(i)))];
                end
            elseif strcmp(obj.method, 'SMO')
                for i=2:length(obj.info.xs(1,:))
                    QP_qCon = [QP_qCon norm(obj.info.xs(:, i)-obj.A)/norm(obj.info.xs(:, i-1)-obj.A)];
                end
                tol = 0.001;
                for i=1:length(obj.info.xs(1,:))
                    cond_check = 0;
                    for j=1:obj.size
                        E = (sum(obj.info.xs(:,i).*obj.y.*obj.KernelMatrix(:, j), 'all')+obj.b)*obj.y(j);
                        if obj.A(j) == 0 && E >= 1-tol
                            cond_check = cond_check+1;
                        elseif obj.A(j) == obj.C &&  E<= 1+tol
                            cond_check = cond_check+1;
                        elseif obj.A(j) < obj.C && obj.A(j) > 0 && E-1<tol
                            cond_check =  cond_check+ 1;
                        end
                    end
                    QP_grad = [QP_grad, cond_check/obj.size];
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
                
        function [H, KernelMatrix] = setH(obj)            
            H = zeros(obj.size, obj.size);
            KernelMatrix = zeros(obj.size, obj.size);
            for i=1:obj.size
                for j=1:obj.size                              
                    H(i,j) = obj.y(i)* obj.y(j)* obj.kernel(obj.X(i,:), obj.X(j,:));
                    KernelMatrix(i,j) = obj.kernel(obj.X(i,:), obj.X(j,:));
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
