%% Reimlementation of the experient in https://uk.mathworks.com/help/stats/fitcsvm.html#bt9w6j6-2
%linear seperable binary classification
clear all
load fisheriris
inds = ~strcmp(species,'setosa');
X = meas(inds,3:4);
y_cat = species(inds);

% X = X(1:50,:);
% y_cat = y_cat(1:50,:);

y = grp2idx(y_cat);

y(y == 1) = -1;
y(y == 2) = 1;
C=1000;


s = svmOpt(X, y, C)

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

%pred = (s.predict(X) == y);

figure, contour(x1_grid,x2_grid, out, [0,0] ,'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(X(:,1),X(:,2),y_cat)
plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
hold off

%outlier

load fisheriris
X_nonlin = meas(:,1:2);
y_nonlin = ones(size(X_nonlin,1),1);
C=10000;
for i=1:length(X_nonlin(1,:))                    
    X_nonlin(:,i) = (X_nonlin(:,i) - mean(X_nonlin(:,i)))/sqrt(var(X_nonlin(:,i)));
end
s_outlier = svmOpt(X_nonlin, y_nonlin, C, 0.05)

n = 300;
x1 = linspace(min(X_nonlin(:,1))-0.5, max(X_nonlin(:,1))+0.5, n+1);
x2 = linspace(min(X_nonlin(:,2))-0.5, max(X_nonlin(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s_outlier.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

figure, contour(x1_grid,x2_grid, out, [0,0], 'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, 'ShowText','on')
plot(X_nonlin(:,1),X_nonlin(:,2), 'k.')
plot(s_outlier.SVs(:,1),s_outlier.SVs(:,2),'ko','MarkerSize',10)
% legend('hyper-plane', 'Support Vecotr')
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

s_nonlin = svmOpt(data3, theclass, C)

n = 300;
x1 = linspace(min(data3(:,1))-0.5, max(data3(:,1))+0.5, n+1);
x2 = linspace(min(data3(:,2))-0.5, max(data3(:,2))+0.5, n+1);
[x1_grid, x2_grid] = meshgrid(x1,x2);
score = s_nonlin.predict([x1_grid(:), x2_grid(:)]);
out = reshape(score,size(x1_grid,1),size(x2_grid,2));

%pred = (s.predict(X) == y);

figure, contour(x1_grid,x2_grid, out, [0,0], 'ShowText','on')
hold on
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], 'k-.', 'ShowText','on')
gscatter(data3(:,1),data3(:,2),theclass)
plot(s_nonlin.SVs(:,1),s_nonlin.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
hold off