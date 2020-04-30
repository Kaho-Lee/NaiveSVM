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

optimizer = 'SR1';
%optimizer = 'ConjugateGrad';

s = svmOpt(X, y,  'linear', C, 0, optimizer)

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
contour(x1_grid,x2_grid, out, [ [1,1], [-1,-1]], '-.', 'ShowText','on')
gscatter(X(:,1),X(:,2),y_cat)
plot(s.SVs(:,1),s.SVs(:,2),'ko','MarkerSize',10)
legend('hyper-plane','margin','-1','1','Support Vector')
hold off
