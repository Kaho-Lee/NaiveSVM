classdef svmOpt
    properties
        H;
        X;
        y;
        size;
        C;
        A;
        L;
        SVs;
        isSV;
        W;
        b;
        kernel;
        numclass;
        outlierFraction;
    end
    
    methods
        function obj = svmOpt(X, y, varargin)
            obj.X = X;
            obj.y = y;
            size_X = size(obj.X);
            obj.size = size_X(1);
            obj.numclass = length(unique(y));
            
            %obj.kernel = @(x1, x2) x1 * (x2.');
            sigma = 1;
            obj.kernel = @(x1, x2) exp(-norm(x1-x2)/(2*sigma));
            
            H = setH(obj);
            obj.H = H;
            
            if nargin==2
                % third parameter does not exist, so default it to something
                obj.C = 0;
            elseif nargin==3
                obj.C = varargin{1};
                obj.outlierFraction = 0;
            elseif nargin==4
                obj.C = 1/(length(obj.y)*varargin{2});
                obj.outlierFraction = varargin{2};
            end
            
                        
            obj.A = zeros(obj.size, 1);
            %quadratic penalty objective function, dual form
            
            obj.L = QudraticPenalty(obj);
            
            obj.A = fit(obj);
            [obj.SVs, obj.isSV] = assignSupportVecotor(obj);
            %obj.W = setW(obj);
            obj.b = setBias(obj);
        end
        
        function aMin = fit(obj)
            mu0 = 1;
            a0 = obj.A;
            [aMin, fMin_QP, nIter_QP, info_QP] = PenaltyAugmented(obj.L, mu0, a0, 'QuadraticPenalty');                       
            disp('finish training')
        end
        
        function pred = predict(obj, X)
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
                disp('2 predict')
                for s=1:length(X)
                    tot_sum = 0;
                    for m=1:obj.size
                       tot_sum = tot_sum + obj.A(m) .* obj.kernel(obj.X(m,:), X(s,:));
                    end
                    logits = [logits, tot_sum-obj.b];
                end
            end
            pred = logits;
        end
        
        
        function L = QudraticPenalty(obj)
            
            if obj.outlierFraction == 0           
                L.f = @(a, mu) 0.5*sum((a * a.').*obj.H, 'all') - sum(a, 'all') ...
                    + (mu/2)*((a.' * obj.y).^2 + sum(max(0, 0-a).^2, 'all') ...
                    + sum(max(0, a-obj.C).^2, 'all') );

                str = '@(a, mu, H, y, C)[';
                for i=1:obj.size
                    df_i = sprintf("0.5*sum(a.'.*H(%d,:), 'all') - 1 + mu*((a.' * y)*y(%d)+min(0, a(%d))+max(0, a(%d)-C));",i,i,i,i);
                    str = strcat(str,df_i);
                end
                str = strcat(str,"];");
                df = str2func(str);
                L.df = @(a, mu) df(a, mu, obj.H, obj.y, obj.C);
            elseif obj.outlierFraction > 0
                disp('2 Quadratic penalty')
                L.f = @(a, mu) 0.5*sum((a * a.').*obj.H, 'all') ...
                    + (mu/2)*((sum(a,'all')-1).^2 + sum(max(0, 0-a).^2, 'all') ...
                    + sum(max(0, a-obj.C).^2, 'all') );
                
                str = '@(a, mu, H, C)[';
                for i=1:obj.size
                    df_i = sprintf("0.5*sum(a.'.*H(%d,:), 'all') + mu*((sum(a,'all')-1)+min(0, a(%d))+max(0, a(%d)-C));",i,i,i);
                    str = strcat(str,df_i);
                end
                str = strcat(str,"];");
                df = str2func(str);
                L.df = @(a, mu) df(a, mu, obj.H, obj.C);
            end
        end
                
        function H = setH(obj)            
            H = zeros(obj.size, obj.size);
            
            for i=1:obj.size
                for j=1:obj.size
%                     H(i,j) = obj.y(i)* obj.y(j)*(obj.X(i,:) * obj.X(j,:).');                                
                    H(i,j) = obj.y(i)* obj.y(j)* obj.kernel(obj.X(i,:), obj.X(j,:));  
                end
            end            
        end
        
        function value = sign(logits)
            logits(logits < -500) = -500;
            logits(logits > 500) = 500;
            value = 1/(1+exp(-logits));
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
                disp('2 bias')
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
                    %break
                end
                b = sum(temp)/sum(obj.isSV);                                
            end
        end
        
    end
end
