%% NC_Project_Fall2025 - Main Execution Script
% This script runs the complete linear regression project
% Dr. Muhammad Tayyab - Numerical Computations Lab

clear; close all; clc;

fprintf('============================================\n');
fprintf('Numerical Computations Lab Project - Fall 2025\n');
fprintf('Linear Regression with Gradient Descent\n');
fprintf('============================================\n\n');

%% Add utils folder to path
addpath('utils');

%% Part 1: Test on Synthetic Data
fprintf('PART 1: Testing on Synthetic Data\n');
fprintf('---------------------------------\n');
test_synthetic;

%% Part 2: Train on Real Data (Square_Footage vs Price)
fprintf('\n\nPART 2: Training on Real Data\n');
fprintf('---------------------------------\n');
train_real;

%% Summary
fprintf('\n\n============================================\n');
fprintf('Project Execution Complete\n');
fprintf('All plots saved to /outputs/ folder\n');
fprintf('============================================\n');

%% Remove utils from path
rmpath('utils');