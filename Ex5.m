clear all
a = 1;
b = 1.5;
Q.f = @(x, mu) (x(1)-a).^2 + 0.5*(x(2)-b).^2 - 1 + (mu/2)*(x(1).^2 + x(2).^2 - 2).^2;
Q.df = @(x, mu) [2*(x(1)-a) + mu*(x(1).^2 + x(2).^2 - 2)*2*x(1);
                     (x(2)-b) + mu*(x(1).^2 + x(2).^2 - 2)*2*x(2)];
Q.d2f = @(x, mu) [2+mu*(6*x(1).^2 + 2*x(2).^2 -4), 4*mu*x(1)*x(2);
                      4*mu*x(1)*x(2), 1+mu*(2*x(1).^2+6*x(2).^2-4)];
QPenalty_visual = @(x,y,a,b, mu) (x-a).^2 + 0.5*(y-b).^2 + (mu/2)*(x.^2 + y.^2 - 2).^2;

LA.f = @(x, mu, v) (x(1)-a).^2 + 0.5*(x(2)-b).^2 - 1 - v*(x(1).^2 + x(2).^2 - 2) + (mu/2)*(x(1).^2 + x(2).^2 - 2).^2;
LA.df = @(x, mu, v) [2*(x(1)-a) - 2*v*x(1) + mu*(x(1).^2 + x(2).^2 - 2)*2*x(1);
                  (x(2)-b) - 2*v*x(2) + mu*(x(1).^2 + x(2).^2 - 2)*2*x(2)];
LA.d2f = @(x, mu, v) [2- 2*v + mu*(6*x(1).^2 + 2*x(2).^2 -4), 4*mu*x(1)*x(2);
                   4*mu*x(1)*x(2), 1- 2*v + mu*(2*x(1).^2+6*x(2).^2-4)];
LA_visual = @(x,y,a,b, mu, v) (x-a).^2 + 0.5*(y-b).^2 + v*(x.^2 + y.^2 - 2) + (mu/2)*(x.^2 + y.^2 - 2).^2; 

%% Exp feasible start point
%Quadratic Penalty
mu0 = 1;
x0 = [1; 1];
[xMin_QP, fMin_QP, nIter_QP, info_QP] = PenaltyAugmented(Q, mu0, x0, 'QuadraticPenalty');
temp_QP = norm(xMin_QP, 2).^2
fMin_QP

%Augmented Lagrangian
mu0 = 1;
v_0 = 1;
c_1 = @(x) x(1).^2 + x(2).^2 - 2;
[xMin_LA, fMin_LA, nIter_LA, info_LA] = PenaltyAugmented(LA, mu0, x0, 'AugmentedLagrangian', v_0, c_1);
temp_LA = norm(xMin_LA, 2).^2
fMin_LA

%Visual
n =300
x = linspace(x0(1)-0.5, x0(1)+0.5, n+1);
y = linspace(x0(2)-0.5, x0(2)+0.5, n+1);
[X, Y] = meshgrid(x,y);
out = QPenalty_visual(X,Y, a, b, mu0);
figure, contour(X,Y,out, 'ShowText','on')
hold on
plot(info_QP.xs(1,:), info_QP.xs(2,:), '.-')
plot(info_LA.xs(1,:), info_LA.xs(2,:), '.-')
txt = sprintf('Trajectories of Quadratic Penalty and Augmented Lagrangian, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('x') 
ylabel('y')
legend('Contour','Quadratic Penalty', 'Augmented Lagrangian')
hold off

QP_qCon = [];

for i=2:length(info_QP.xs(1,:))
    QP_qCon = [QP_qCon norm(info_QP.xs(:, i)-xMin_QP)/norm(info_QP.xs(:, i-1)-xMin_QP)];
end

QP_grad = [];
for i = 1: length(info_QP.mus)
    QP_grad = [QP_grad norm(Q.df(info_QP.xs(:,i), info_QP.mus(i)))];
end

LA_qCon = [];
for i=2:length(info_LA.xs(1,:))
    LA_qCon = [LA_qCon norm(info_LA.xs(:, i)-xMin_LA)/norm(info_LA.xs(:, i-1)-xMin_LA)];
end

LA_x_v = []
for i = 1:length(info_LA.xs(1,:))
    
    temp = norm(info_LA.xs(:, i) - xMin_LA)/norm(info_LA.vks(i) - info_LA.vks(length(info_LA.vks)));
    LA_x_v = [LA_x_v temp];
end
LA_v_v = []
for i = 2:length(info_LA.xs(1,:))
    temp = norm(info_LA.vks(i) - info_LA.vks(length(info_LA.vks)))/norm(info_LA.vks(i-1) - info_LA.vks(length(info_LA.vks)));
    LA_v_v = [LA_v_v temp];
end

figure, plot(QP_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)
hold on
ylim([0 inf])
plot(QP_grad, '.-', 'linewidth', 2, 'MarkerSize',12)
txt = sprintf('Convergence of Quadratic Penalty, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}', '||\nabla Q||_{2}')
hold off

figure, plot(LA_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)
hold on
plot(LA_x_v, '.-', 'linewidth', 2, 'MarkerSize',12)
plot(LA_v_v, '.-', 'linewidth', 2, 'MarkerSize',12)
ylim([0 inf])
txt = sprintf('Convergence of Augmented Lagrangians, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}', '||x_{k} - x^{*}||_{2}/||v_{k} - v^{*}||_{2}', '||v_{k} - v^{*}||_{2}/||v_{k-1} - v^{*}||_{2}')
hold off

%% Exp infeasible start point
%Quadratic Penalty
mu0 = 1;
x0 = [-1.5; -1];
[xMin_QP_if, fMin_QP_if, nIter_QP_if, info_QP_if] = PenaltyAugmented(Q, mu0, x0, 'QuadraticPenalty');
temp_QP_if = norm(xMin_QP_if, 2).^2


%Augmented Lagrangian
mu = 10;
v_0 = 1;
c_1 = @(x) x(1).^2 + x(2).^2 - 2;
[xMin_LA_if, fMin_LA_if, nIter_LA_if, info_LA_if] = PenaltyAugmented(LA, mu0, x0, 'AugmentedLagrangian', v_0, c_1);
temp_LA_if = norm(xMin_LA, 2).^2


%Visual
n =300
x = linspace(x0(1)-0.5, 1.5, n+1);
y = linspace(x0(2)-0.5, 2, n+1);
[X, Y] = meshgrid(x,y);
out = QPenalty_visual(X,Y, a, b, mu0);
figure, contour(X,Y,out, 'ShowText','on')
hold on
plot(info_QP_if.xs(1,:), info_QP_if.xs(2,:), '.-')
plot(info_LA_if.xs(1,:), info_LA_if.xs(2,:), '.-')
txt = sprintf('Trajectories of Quadratic Penalty and Augmented Lagrangian, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('x') 
ylabel('y')
legend('Contour','Quadratic Penalty', 'Augmented Lagrangian')
hold off

QP_qCon = [];
for i=2:length(info_QP_if.xs(1,:))
    QP_qCon = [QP_qCon norm(info_QP_if.xs(:, i)-xMin_QP_if)/norm(info_QP_if.xs(:, i-1)-xMin_QP_if)];
end

QP_grad = [];
for i = 1: length(info_QP_if.mus)
    QP_grad = [QP_grad norm(Q.df(info_QP_if.xs(:,i), info_QP_if.mus(i)))];
end


LA_qCon = [];
for i=2:length(info_LA_if.xs(1,:))
    LA_qCon = [LA_qCon norm(info_LA_if.xs(:, i)-xMin_LA_if)/norm(info_LA_if.xs(:, i-1)-xMin_LA_if)];
end

LA_x_v = []
for i = 1:length(info_LA_if.xs(1,:))
    
    temp = norm(info_LA_if.xs(:, i) - xMin_LA_if)/norm(info_LA_if.vks(i) - info_LA_if.vks(length(info_LA_if.vks)));
    LA_x_v = [LA_x_v temp];
end
LA_v_v = []
for i = 2:length(info_LA_if.xs(1,:))
    temp = norm(info_LA_if.vks(i) - info_LA_if.vks(length(info_LA_if.vks)))/norm(info_LA_if.vks(i-1) - info_LA_if.vks(length(info_LA_if.vks)));
    LA_v_v = [LA_v_v temp];
end

figure, plot(QP_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)
hold on
ylim([0 inf])
plot(QP_grad, '.-', 'linewidth', 2, 'MarkerSize',12)
txt = sprintf('Convergence of Quadratic Penalty, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}', '||\nabla Q||_{2}')
hold off

figure, plot(LA_qCon, '.-', 'linewidth', 2, 'MarkerSize',12)
hold on
plot(LA_x_v, '.-', 'linewidth', 2, 'MarkerSize',12)
plot(LA_v_v, '.-', 'linewidth', 2, 'MarkerSize',12)
ylim([0 inf])
txt = sprintf('Convergence of Augmented Lagrangians, Starting from (%.2f, %.2f)', x0);
title(txt)
xlabel('Iteration k')
legend('||x_{k} - x^{*}||_{2}/||x_{k-1} - x^{*}||_{2}', '||x_{k} - x^{*}||_{2}/||v_{k} - v^{*}||_{2}', '||v_{k} - v^{*}||_{2}/||v_{k-1} - v^{*}||_{2}')
hold off

