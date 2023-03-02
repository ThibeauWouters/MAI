%%%%%%%%%%%%%%%%%%%%%%%%%
% Session 1, exercise 2 %
%%%%%%%%%%%%%%%%%%%%%%%%%

% Some data X1, X2, Y 
X1_data = 2*pi*rand(1000, 1)-pi;
X2_data = 2*pi*rand(1000, 1)-pi;
Y_data = cos(X1_data)-sin(X2_data);

% define an interpolation function for your data
f = scatteredInterpolant(X1_data, X2_data, Y_data);

% create a regular grid and evaluate the interpolations on it
x1 = linspace(min(X1_data), max(X1_data), 1000);
x2 = linspace(min(X2_data), max(X2_data), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
hold on;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Some 3D function');
xlabel('X 1');
ylabel('X 2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed

%plot3(X1_data, X2_data, Y_data,'.','MarkerSize',15)