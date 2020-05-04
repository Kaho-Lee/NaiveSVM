function [A, b, info] = SMO( y, C, tol, max_passes, dataSize, KernelMat)
% mainly follow the pseudocode in http://cs229.stanford.edu/materials/smo.pdf
    A= zeros(dataSize,1);
    b = 0;
    passes = 0;
    info.xs = A;
    while passes < max_passes
        num_changed_alphas = 0;
        for i=1:dataSize
            E_i = sum(A.*y.*KernelMat(:, i), 'all')+b - y(i);
            if (y(i)*E_i < -tol && A(i)<C) || ...
                    (y(i)*E_i > tol && A(i)>0)
                j = i;
                while j == i
                    j = randsample(dataSize,1);
                    if j ~= i
                        break;
                    end
                end
                E_j = sum(A.*y.*KernelMat(:, j), 'all')+b - y(j);
                a_i_old = A(i);
                a_j_old = A(j);

                if y(i)~=y(j)
                    L = max(0, A(j)-A(i));
                    H = min(C, C+A(j)-A(i));
                else
                    L = max(0, A(i)+A(j)-C);
                    H = min(C, A(i)+A(j));
                end

                if L==H
                    continue
                end

                eta = 2*KernelMat(i,j)-KernelMat(i,i)-KernelMat(j,j);
                if eta>=0
                    continue
                end

                A(j) = A(j) - (y(j)*(E_i-E_j))/eta;
                if A(j) > H
                    A(j) = H;
                elseif A(j) <L
                    A(j) = L;
                end
                if abs(A(j) - a_j_old)< 1e-4
                    continue
                end
                A(i) = A(i) + y(i)*y(j)*(a_j_old - A(j));
                
                b1 = b - E_i - y(i)*(A(i)-a_i_old)*KernelMat(i,i)-y(j)*(A(j)-a_j_old)*KernelMat(i,j);
                b2 = b - E_j - y(i)*(A(i)-a_i_old)*KernelMat(i,j)-y(j)*(A(j)-a_j_old)*KernelMat(j,j);
                
                if A(i)>0 && A(i)<C
                    b = b1;
                elseif A(j)>0 && A(j)<C
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                num_changed_alphas = num_changed_alphas + 1;                    
            end
        end
        
        if num_changed_alphas == 0
            passes =passes + 1;
        else
            passes = 0;
            info.xs = [info.xs, A];
        end
    end
    disp('finish training')
end