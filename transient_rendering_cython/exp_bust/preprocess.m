clear; clc; close all;

addpath('../exp_bunny');
folder = 'setup/';


item = 'bust';
transient_location = [folder item '_transient.mat'];
cnlos_location = [folder 'cnlos_' item '_threshold.obj'];
space_carving_location = [folder 'space_carving_' item '.obj'];

compute_init_mesh(transient_location, cnlos_location, 1*10^-3);
compute_space_carving_mesh(transient_location, space_carving_location);


