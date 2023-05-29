%%%%% 3D plotting
%%% Plot the original data points and mesh in 3D

% Create 3D plot of the training data
F = scatteredInterpolant(X1, X2, Tnew);

% Define a grid of points at which to evaluate the surface
[Xq,Yq] = meshgrid(linspace(0, 1, 100));

% Evaluate the surface at the grid points
Zq = F(Xq, Yq);

%figure;
%plot3(X1, X2, Tnew,'o');
% Plot the interpolated surface using mesh
%mesh(Xq, Yq, Zq);

%%% Code from Toledo
f = scatteredInterpolant(train_X1, train_X2, train_Tnew);

% create a regular grid and evaluate the interpolations on it
x1 = linspace(min(train_X1), max(train_X1), 1000);
x2 = linspace(min(train_X2), max(train_X2), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Distribution of training data');
xlabel('X1');
ylabel('X2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(train_X1, train_X2, train_Tnew,'.','MarkerSize',15)
hold off;