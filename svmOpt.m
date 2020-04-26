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
    end
    
    methods
        function obj = svmOpt(X, y, varargin)
            obj.X = X;
            obj.y = y;
            size_X = size(obj.X);
            obj.size = size_X(1);
            
            obj.kernel = @(x1, x2) x1 * (x2.');
            %var = 1;
            %obj.kernel = @(x1, x2) exp(-norm(x1-x2)/(2*var));
            
            H = setH(obj);
            obj.H = H;
            
            if nargin==2
                % third parameter does not exist, so default it to something
                obj.C = 0;
            else
                obj.C = varargin{1};
            end
                        
            obj.A = zeros(obj.size, 1);
            %quadratic penalty objective function, dual form
            
            obj.L = QudraticPenalty(obj);
            %obj.L = AugmentedLagrangian(obj);
            
            obj.A = fit(obj);
            [obj.SVs, obj.isSV] = assignSupportVecotor(obj);
            %obj.W = setW(obj);
            obj.b = setBias(obj);
        end
        
        function aMin = fit(obj)
            mu0 = 1;
            a0 = obj.A;
            [aMin, fMin_QP, nIter_QP, info_QP] = PenaltyAugmented(obj.L, mu0, a0, 'QuadraticPenalty');
                       
            %v_0 = [1 1 1];
            %c_1 = @(a) [a.' * obj.y, sum(max(0, 0-a), 'all'), sum(max(0, a-obj.C), 'all')];
            %[aMin, fMin_LA, nIter_LA, info_LA] = PenaltyAugmented(obj.L, mu0, a0, 'AugmentedLagrangian', v_0, c_1);
            disp('finish training')
        end
        
        function pred = predict(obj, X)
            logits = [];
            
            for s=1:length(X)
                tot_sum = 0;
                for m=1:obj.size
                   tot_sum = tot_sum + obj.A(m) .* obj.y(m) .* obj.kernel(obj.X(m,:), X(s,:));
                end
                logits = [logits, tot_sum+obj.b];
            end
            pred = logits;
        end
        
        function L = AugmentedLagrangian(obj)
            %Augmented Lagrangian
            L.f = @(a, mu, v) 0.5*sum((a * a.').*obj.H, 'all') - sum(a, 'all') ...
                - v(1)*a.' * obj.y - v(2)*sum(max(0, a), 'all') - v(3)*sum(max(0, obj.C-a), 'all') ...
                + (mu/2)*((a.' * obj.y).^2 + sum(max(0, 0-a).^2, 'all') ...
                + sum(max(0, a-obj.C).^2, 'all') );
            str = '@(a, mu, v, H, y, C)[';
            for i=1:obj.size
                df_i = sprintf("0.5*sum(a.'.*H(%d,:), 'all')-1-v(1)*y(%d)+v(2)*heaviside(a(%d))-v(3)*heaviside(C-a(%d))+mu*((a.' * y)*y(%d)+min(0, a(%d))+max(0, a(%d)-C));",i,i,i,i,i,i,i);
                str = strcat(str,df_i);
            end
            str = strcat(str,"];");
            df = str2func(str);
            L.df = @(a, mu, v) df(a, mu,v, obj.H, obj.y, obj.C);
        end
        
        function L = QudraticPenalty(obj)
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
        
        function W = setW(obj)
            
            a_y = obj.A .* obj.y;
            W = a_y .* obj.X;
            W = sum(W, 1);
        end
        
        function b = setBias(obj)            
%             a_y_x_m = sum( obj.A .* obj.y .* obj.kernel(obj.X, x_s), 1); %1x2 w = sum(a*y*X)
                                   
            y_s = obj.isSV .* obj.y; % 100x1
            
            x_s = (obj.isSV .* obj.X); %2x100
            %temp = sum( obj.A .* obj.y .* obj.kernel(obj.X, x_s), 1); %1x100 w = sum(a*y*x_n*x_s)
            temp = [];
            for s=1:obj.size
                tot_sum = 0;
                for m=1:obj.size
                   tot_sum = tot_sum + obj.A(m) .* obj.y(m) .* obj.kernel(obj.X(m,:), x_s(s,:));
                end
                temp = [temp tot_sum];
            end
            
%             temp = a_y_x_m * x_s; %1x100
            b = sum(y_s - temp.')/sum(obj.isSV);             
        end
        
    end
end
