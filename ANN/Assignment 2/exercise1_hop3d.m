clear
clc
close all

%%%%%%%%%%%
% exercise1_hop3d.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%

savedata = true;
if savedata
    % Create a new, empty CSV file for saving the data
    filename = 'hop_3D_trajectories.csv';
    % Clear the contents of the CSV file
    fclose(fopen(filename, 'w'));
end

% Define the x, y, z ranges for the grid of points
lower_bound = -0.75;
upper_bound =  0.75;
nb_of_points = 5;
x_range = linspace(lower_bound, upper_bound, nb_of_points);
y_range = linspace(lower_bound, upper_bound, nb_of_points);
z_range = linspace(lower_bound, upper_bound, nb_of_points);

% Create the mesh of grid points using meshgrid
[x, y, z] = meshgrid(x_range, y_range, z_range);

% Define patterns and network:
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);

% define number of steps to simulate the network
number_of_epochs = 1000;

tic;
for i = 1:numel(x)
    x_val = x(i);
    y_val = y(i);
    z_val = z(i);
    % Define starting point
    start = {[x_val; y_val; z_val]};
    [path, Pf, Af] = sim(net, {1 number_of_epochs}, {}, start);
    record = [cell2mat(start) cell2mat(path)];
    if savedata
        writematrix(record, filename, 'WriteMode', 'append');
    end
end
timed = toc;
disp("Time:");
disp(timed);