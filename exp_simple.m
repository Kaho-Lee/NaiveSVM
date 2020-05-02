%% Reimlementation of the experient in https://uk.mathworks.com/help/stats/fitcsvm.html#bt9w6j6-2
%linear seperable binary classification
clear all
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y_cat = species(inds);

X_trun = X(1:50,:);
y_cat_trun = y_cat(1:50,:);

y_trun = grp2idx(y_cat_trun);

y_trun(y_trun == 1) = -1;
y_trun(y_trun == 2) = 1;
C=1000;

optimizer = 'SR1';
%optimizer = 'ConjugateGrad';
% for i=1:length(X(1,:))                    
%     X_trun(:,i) = (X_trun(:,i) - mean(X_trun(:,i)))/std(X_trun(:,i));
% end

s = SVM_Opt_model(X_trun, y_trun,  'RBF', C, 0.5, optimizer)

constraint_ya = y_trun.' * s.A;
constraint_a_minus_c = s.A - C;


X_test = X(1:55,:);
y_cat_test = y_cat(1:55,:);

y_test = grp2idx(y_cat_test);

y_test(y_test == 1) = -1;
y_test(y_test == 2) = 1;

for i=1:length(X(1,:))                    
    X_test(:,i) = (X_test(:,i) - mean(X_test(:,i)))/std(X_test(:,i));
end

n = 300;
x1 = linspace(min(X_test(:,1))-0.5, max(X_test(:,1))+0.5, n+1);
x2 = linspace(min(X_test(:,2))-0.5, max(X_test(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

%pred = (s.predict(X) == y);

figure, contour(x1_grid,x2_grid, out, 'ShowText','on')
hold on
% contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], '-.', 'ShowText','on')
gscatter(X_test(:,1),X_test(:,2),y_cat_test)
%plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
legend('contour','-1','1','Support Vector')
hold off

logits = s.predict(X_test);

[TPR_lst, FPR_lst] = GenRoc(logits, y_test);
figure, plot(FPR_lst, TPR_lst)
hold on
plot(FPR_lst, FPR_lst)
legend('roc', 'baseline')
hold off

function [TPR, FPR] = GetStat(prob, mask, threshold, target)
    temp = zeros(length(prob), 1);
    for i=1:length(prob)
        if prob(i) > threshold
            temp(i) = -1;
        else
            temp(i) = 1;
        end
    end
    
    TP = zeros(length(prob), 1);
    FP = zeros(length(prob), 1);
    TN = zeros(length(prob), 1);
    FN = zeros(length(prob), 1);
    for i=1:length(prob)
        if temp(i)==target && mask(i) == target
            TP(i) = 1;
        elseif temp(i)==target && mask(i) ~= target
            FP(i) = 1;
        elseif temp(i)~=target && mask(i) ~= target
            TN(i) = 1;
        elseif temp(i)~=target && mask(i) == target
            FN(i) = 1;
        end
    end
    TP = sum(TP, 'all');
    FP = sum(FP, 'all');
    TN = sum(TN, 'all');
    FN = sum(FN, 'all');
    
    TPR = TP/(TP+FN);
    FPR = FP/(FP+TN);
end

function value = sigmoid( logits)
%     logits(logits < -500) = -500;
%     logits(logits > 500) = 500;
    value = 1./(1+exp(-logits));
end

function [TPR_lst, FPR_lst] = GenRoc(logits, mask)
    TPR_lst = [];
    FPR_lst = [];
    size(logits)
    prob = sigmoid( logits);
    range = [0.5:0.01:1];
    for i= 1:length(range)
        [TPR, FPR] = GetStat(prob, mask, range(i), 1);
        TPR_lst = [TPR_lst TPR];
        FPR_lst = [FPR_lst FPR];
    end
end


%linear seperable binary classification
% clear all
% load fisheriris
% inds = ~strcmp(species,'setosa');
% X = meas(inds,3:4);
% y_cat = species(inds);
% 
% for i=1:length(X(1,:))                    
%     X(:,i) = (X(:,i) - mean(X(:,i)))/std(X(:,i));
% end
% 
% % X = X(1:50,:);
% % y_cat = y_cat(1:50,:);
% 
% y = grp2idx(y_cat);
% 
% y(y == 1) = -1;
% y(y == 2) = 1;
% C=10;
% % X = X(1:90,:);
% % y = y(1:90);
% % y_cat = y_cat(1:90);
% 
% optimizer = 'SR1';
% %optimizer = 'ConjugateGrad';
% 
% s = SVM_Opt_model(X, y, 'linear', C, 0, optimizer)
% 
% a = zeros( 100, 1);
% mu = 1;
% constraint_ya = y.' * s.A;
% constraint_a_minus_c = s.A - C;
% 
% n = 300;
% x1 = linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, n+1);
% x2 = linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, n+1);
% [x1_grid, x2_grid] = meshgrid(x1,x2);
% score = s.predict([x1_grid(:), x2_grid(:)]);
% out = reshape(score,size(x1_grid,1),size(x2_grid,2));
% 
% %pred = (s.predict(X) == y);
% 
% figure, contour(x1_grid,x2_grid, out, [0,0] ,'ShowText','on')
% hold on
% contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
% gscatter(X(:,1),X(:,2),y_cat)
% plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
% legend('hyper-plane','margin','-1','1','Support Vector')
% hold off