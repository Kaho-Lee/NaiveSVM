clear all
rng(1)
filename = 'data/ionosphere.data';

fid = fopen(filename);
data_format = '';
for i=1:34
    data_format = strcat(data_format, '%f');
end
data_format = strcat(data_format, '%s')
data=textscan(fid,data_format,'delimiter',',');
fclose(fid);

data_y_cat = data{35};
data_x = zeros(351,34);

data_y = grp2idx(data_y_cat);
data_y(data_y == 1) = -1; %good - negative (majority)
data_y(data_y == 2) = 1; %bad - positive (anomaly)
for i=1:34
    data_x(:,i) = [data{i}];
end
B_index = find(data_y==1);
G_index = find(data_y==-1);
negative_x = data_x(G_index, :);
negative_y = data_y(G_index, :);
positive_x = data_x(B_index, :);
positive_y = data_y(B_index, :);

% train_x = [negative_x(1:ceil(length(G_index)/2), :); positive_x(1:ceil(length(B_index)/2), :)];
% train_y = [negative_y(1:ceil(length(G_index)/2), :); positive_y(1:ceil(length(B_index)/2), :)];
% test_x = [negative_x(ceil(length(G_index)/2)+1:length(G_index), :); positive_x(ceil(length(B_index)/2)+1:length(B_index), :)];
% test_y = [negative_y(ceil(length(G_index)/2)+1:length(G_index), :); positive_y(ceil(length(B_index)/2)+1:length(B_index), :)];

selectPositiveData = randperm(length(B_index), 5);
train_x = [negative_x(1:ceil(length(G_index)/2), :); positive_x(selectPositiveData, :)];
train_y = [negative_y(1:ceil(length(G_index)/2)); positive_y(selectPositiveData)];

selectPositiveData = randperm(length(B_index), 5);
test_x = [negative_x(ceil(length(G_index)/2)+1:length(G_index), :); positive_x(selectPositiveData, :)];
test_y = [negative_y(ceil(length(G_index)/2)+1:length(G_index)); positive_y(selectPositiveData)];

C=10;
optimizer = 'SMO';
s_class_SMO = SVM_Opt_model(train_x, train_y,  'RBF', C, 0, 'SMO', optimizer)
logits = s_class_SMO.predict(test_x);
[TPR_lst_SMO, FPR_lst_SMO] = GenRoc(logits, test_y);
area = AUC(TPR_lst_SMO, FPR_lst_SMO);
txt1 = sprintf('SMO: AUC=%.4f, CPUTime=%.3f s', area, s_class_SMO.e);

optimizer = 'BFGS';
s_class_BFGS = SVM_Opt_model(train_x, train_y,  'RBF', C, 0, 'QuadraticPenalty', optimizer)
logits = s_class_BFGS.predict(test_x);
[TPR_lst_BFGS, FPR_lst_BFGS] = GenRoc(logits, test_y);
area = AUC(TPR_lst_BFGS, FPR_lst_BFGS);
txt2 = sprintf('BFGS: AUC=%.4f, CPUTime=%.3f s', area, s_class_BFGS.e);

template = [0:0.05:1];
figure, plot(FPR_lst_SMO, TPR_lst_SMO, 'linewidth', 2, 'MarkerSize',12)
hold on
plot(FPR_lst_BFGS, TPR_lst_BFGS, 'linewidth', 2, 'MarkerSize',12)
plot(template, template, 'linewidth', 2, 'MarkerSize',12)
legend(txt1,txt2, 'baseline')
title('ROC Curve of Binary Classification');
hold off

figure, plot(s_class_BFGS.QP_qCon, 'o-', 'linewidth', 2, 'MarkerSize',7)
hold on
plot(s_class_BFGS.QP_grad, 'o-', 'linewidth', 2, 'MarkerSize',7)
ylim([0 inf])
txt = sprintf('Convergence Analysis: BFGS');
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}', '||\nabla Q||_{2}')
hold off

figure, plot(s_class_SMO.QP_qCon, '*-', 'linewidth', 2, 'MarkerSize',4)
hold on
plot(s_class_SMO.QP_grad, '*-', 'linewidth', 2, 'MarkerSize',4)
ylim([0 inf])
txt = sprintf('Convergence Analysis: SMO');
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}', 'Fraction: KKT Dual-Complementarity')
hold off

Y = tsne(data_x);
figure, gscatter(Y(:,1),Y(:,2),data_y)

Y = tsne(test_x);
figure, gscatter(Y(:,1),Y(:,2),test_y)

function [TPR, FPR] = GetStat(prob, mask, threshold, target)
    temp = zeros(length(prob), 1);
    for i=1:length(prob)
        if prob(i) > threshold
            temp(i) = 1;
        else
            temp(i) = -1;
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
    range = [0:0.05:1];
    for i= 1:length(range)
        [TPR, FPR] = GetStat(prob, mask, range(i), 1);
        TPR_lst = [TPR_lst TPR];
        FPR_lst = [FPR_lst FPR];
    end
end

function area = AUC(TPR_lst, FPR_lst)
    area = 0;
    for i =1:length(TPR_lst)-1
        slice = abs(FPR_lst(i+1)-FPR_lst(i))*(TPR_lst(i)+TPR_lst(i+1))*0.5;
        area = area + slice;
    end       
end
