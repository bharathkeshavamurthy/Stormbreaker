% TESTDEG2UTM This script will run the function deg2utm with some dummy
% inputs to make sure the resulted (x, y) are what we expect.
%
% Yaguang Zhang, Purdue, 10/05/2017

clear; clc; close all; dbstop if error;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd(fullfile('..', '..')); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'ComputeAntGainTests');

%% Before Processing the Data

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

%% Dummy Data to Use

lat = [38.982217; 38.983182; 38.982014];
lon = [-76.484987; -76.486261; -76.487045];

%% Tests

[x,y, ~] = deg2utm(lat,lon);

hDebugFig = figure('units','normalized', ...
    'outerposition',[0.1 0.2 0.8 0.6]);
subplot(1,2,1);
hold on; plot(lon, lat, 'rx-'); plot_google_map('MapType', 'satellite');
xlabel('Lon'); ylabel('Lat');
subplot(1,2,2); plot(x, y, 'rx-'); xlabel('x'); ylabel('y'); axis equal;

% Save the plot.
curAbsPathToSavePlot = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
    ['testDeg2Utm']);
saveas(hDebugFig, [curAbsPathToSavePlot, '.png']);
saveas(hDebugFig, [curAbsPathToSavePlot, '.fig']);

% EOF