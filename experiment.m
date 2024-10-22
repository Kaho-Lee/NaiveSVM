%% Reimlementation of the experient in https://uk.mathworks.com/help/stats/fitcsvm.html#bt9w6j6-2
%linear seperable binary classification
clear all
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y_cat = species(inds);

% for i=1:length(X(1,:))                    
%     X(:,i) = (X(:,i) - mean(X(:,i)))/std(X(:,i));
% end

% X = X(1:60,:);
% y_cat = y_cat(1:);

y = grp2idx(y_cat);

y(y == 1) = -1;
y(y == 2) = 1;
C=100;

optimizer = 'BFGS';
s = SVM_Opt_model(X, y, 'linear', C, 0, 'QuadraticPenalty', optimizer)

a = zeros( 100, 1);
mu = 1;
constraint_ya = y.' * s.A;
constraint_a_minus_c = s.A - C;

n = 300;
x1 = linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, n+1);
x2 = linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

figure
subplot(1,2,1), contour(x1_grid,x2_grid, out, [0,0] ,'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(X(:,1),X(:,2),y_cat)
plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
title('BFGS')
hold off

optimizer = 'SMO';
s = SVM_Opt_model(X, y, 'linear', C, 0, 'SMO', optimizer)

a = zeros( 100, 1);
mu = 1;
constraint_ya = y.' * s.A;
constraint_a_minus_c = s.A - C;

n = 300;
x1 = linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, n+1);
x2 = linspace(min(X(:,2))-0.5, max(X(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

subplot(1,2,2), contour(x1_grid,x2_grid, out, [0,0] ,'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(X(:,1),X(:,2),y_cat)
plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
title('SMO')
hold off



%% Experiment in https://uk.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html
rng(1); % For reproducibility
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points

data3 = [data1;data2];
theclass = ones(200,1);
theclass(1:100) = -1;

for i=1:length(data3(1,:))                    
    data3(:,i) = (data3(:,i) - mean(data3(:,i)))/std(data3(:,i));
end

optimizer = 'BFGS';
s_nonlin = SVM_Opt_model(data3, theclass, 'RBF', C, 0, 'QuadraticPenalty', optimizer)

n = 300;
x1 = linspace(min(data3(:,1))-0.5, max(data3(:,1))+0.5, n+1);
x2 = linspace(min(data3(:,2))-0.5, max(data3(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s_nonlin.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

figure, 
subplot(1,2,1), contour(x1_grid,x2_grid, out, [0,0], 'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(data3(:,1),data3(:,2),theclass)
plot(s_nonlin.SVs(:,1),s_nonlin.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
title('BFGS')
hold off


s_nonlin = SVM_Opt_model(data3, theclass, 'RBF', C, 0, 'SMO', 'SMO')

n = 300;
x1 = linspace(min(data3(:,1))-0.5, max(data3(:,1))+0.5, n+1);
x2 = linspace(min(data3(:,2))-0.5, max(data3(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s_nonlin.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

 
subplot(1,2,2), contour(x1_grid,x2_grid, out, [0,0], 'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(data3(:,1),data3(:,2),theclass)
plot(s_nonlin.SVs(:,1),s_nonlin.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
title('SMO')
hold off


%% Outlier:Experiment in https://uk.mathworks.com/help/stats/fitcsvm.html#bt9w6j6-2

load fisheriris
X_nonlin = meas(:,1:2);
y_nonlin = ones(size(X_nonlin,1),1);
C=100;
for i=1:length(X_nonlin(1,:))                    
    X_nonlin(:,i) = (X_nonlin(:,i) - mean(X_nonlin(:,i)))/std(X_nonlin(:,i));
end
s_outlier = SVM_Opt_model(X_nonlin, y_nonlin, 'RBF', C, 0.05, 'QuadraticPenalty', optimizer)

n = 300;
x1 = linspace(min(X_nonlin(:,1))-0.5, max(X_nonlin(:,1))+0.5, n+1);
x2 = linspace(min(X_nonlin(:,2))-0.5, max(X_nonlin(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s_outlier.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

figure, contour(x1_grid,x2_grid, out, 'ShowText','on')
hold on
plot(X_nonlin(:,1),X_nonlin(:,2), 'k.')
if ~isempty(s_outlier.SVs)
    plot(s_outlier.SVs(:,1),s_outlier.SVs(:,2),'ko','MarkerSize',10)
end
legend('Logit Value Contour','Observations', 'Support Vecotrs')
hold off