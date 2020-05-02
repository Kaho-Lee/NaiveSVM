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

train_x = [negative_x(1:ceil(length(G_index)/2), :); positive_x(1:5, :)];
train_y = [negative_y(1:ceil(length(G_index)/2)); positive_y(1:5)];
% test_x = [negative_x(ceil(length(G_index)/2)+1:length(G_index), :); positive_x(ceil(length(B_index)/2)+1:length(B_index), :)];
% test_y = [negative_y(ceil(length(G_index)/2)+1:length(G_index), :); positive_y(ceil(length(B_index)/2)+1:length(B_index), :)];

test_x = [negative_x(ceil(length(G_index)/2)+1:length(G_index), :); positive_x(6:10, :)];
test_y = [negative_y(ceil(length(G_index)/2)+1:length(G_index)); positive_y(6:10)];

% for i=1:length(train_x(1,:))
%     if std(train_x(:,i)) >0
%         train_x(:,i) = (train_x(:,i) - mean(train_x(:,i)))/std(train_x(:,i));
%     else
%         continue;
%     end
% end
% 
% for i=1:length(test_x(1,:))
%     if std(test_x(:,i)) >0
%         test_x(:,i) = (test_x(:,i) - mean(test_x(:,i)))/std(test_x(:,i));
%     else
%         continue
%     end
% end

C=10;
optimizer = 'ConjugateGrad';
s_class_PR = SVM_Opt_model(train_x, train_y,  'RBF', C, 0, optimizer)
logits = s_class_PR.predict(test_x);
[TPR_lst_PR, FPR_lst_PR] = GenRoc(logits, test_y);
area = AUC(TPR_lst_PR, FPR_lst_PR);
txt1 = sprintf('PR: AUC=%.4f', area);

optimizer = 'SR1';
s_class_SR1 = SVM_Opt_model(train_x, train_y,  'RBF', C, 0, optimizer)
logits = s_class_SR1.predict(test_x);
[TPR_lst_SR1, FPR_lst_SR1] = GenRoc(logits, test_y);
area = AUC(TPR_lst_SR1, FPR_lst_SR1);
txt2 = sprintf('SR1: AUC=%.4f', area);

template = [0:0.05:1];


figure, plot(FPR_lst_PR, TPR_lst_PR, 'linewidth', 2, 'MarkerSize',12)
hold on
plot(FPR_lst_SR1, TPR_lst_SR1, 'linewidth', 2, 'MarkerSize',12)
plot(template, template, 'linewidth', 2, 'MarkerSize',12)
legend(txt1,txt2, 'baseline')
title('ROC Curve of Binary Classification');
hold off

figure, 
subplot(1,2,1);
plot(s_class_SR1.QP_grad, '.-', 'linewidth', 2, 'MarkerSize',12)
txt = sprintf('SR1: ||\\nabla Q||_{2}');
title(txt)
subplot(1,2,2);
plot(s_class_PR.QP_grad, '.-', 'linewidth', 2, 'MarkerSize',12)
txt = sprintf('PR: ||\\nabla Q||_{2}');
title(txt)


figure, plot(s_class_SR1.QP_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)
hold on
ylim([0 inf])
plot(s_class_PR.QP_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)

txt = sprintf('Convergence Analysis: ||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}');
title(txt)
xlabel('Iteration k')
legend('SR1', 'PR')
hold off

% Y = tsne(data_x);
% figure, gscatter(Y(:,1),Y(:,2),data_y_cat)

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
