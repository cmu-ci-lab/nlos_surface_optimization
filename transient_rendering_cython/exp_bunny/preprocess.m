clear; clc; close all;

folder = 'setup/';

transient_location = [folder 'bunny_transient.mat'];
cnlos_location = [folder 'cnlos_bunny_threshold.obj'];
space_carving_location = [folder 'space_carving_bunny.obj'];

compute_init_mesh(transient_location, cnlos_location);
compute_space_carving_mesh(transient_location, space_carving_location);


