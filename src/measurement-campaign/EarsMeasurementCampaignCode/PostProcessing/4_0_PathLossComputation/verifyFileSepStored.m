% VERIFYFILESEPSTORED Verify the file paths stored in
% pathLossesWithGpsInfo.mat is compatiable with the machine running the
% post processing program.
%
% All the '/' and '\' will be replaced with the filesep for the machine
% running this script, and the results will overwrite the old ones in
% pathLossesWithGpsInfo.mat.
%
% Yaguang Zhang, Purdue, 10/24/2017

clear; clc; close all; dbstop if error;

%% Configurations

warning('on');

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation');
ABS_PATH_TO_PATH_LOSSES_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', ...
    'pathLossesWithGpsInfo.mat');

%% Before Processing the Data

disp(' --------------------- ')
disp('  verifyFileSepStored ')
disp(' --------------------- ')

%% Get Info for Measurement Data Files and Calibration Polynomials

disp(' ')
disp('    Loading results from: ')
disp('      - evalPathLosses.m')

assert(exist(ABS_PATH_TO_PATH_LOSSES_FILE, 'file')==2, ...
    'Couldn''t find pathLossesWithGpsInfo.mat! Please run PostProcessing/4_0_PathLossComputation/evalPathLosses.m first.');

% The data have been processed before and the result files have been found.
disp('    Found all .mat files required.');
disp('        Loading the results...')

% Get 'pathLossesWithGpsInfo', 'relPathsOutFilesUnderDataFolder', and
% 'maxMeasurablePathLossInfo'.
load(ABS_PATH_TO_PATH_LOSSES_FILE);

disp('    Done!')

%% Makesure the FileSep is Correct

disp(' ')
disp('    Checking filesep in relPathsOutFilesUnderDataFolder... ')

relPathsOutFilesUnderDataFolderCell ...
    = arrayfun(@(p) strrep(strrep(p, '/', filesep), '\', filesep), ...
    relPathsOutFilesUnderDataFolder, 'UniformOutput', false);
relPathsOutFilesUnderDataFolder ...
    = [relPathsOutFilesUnderDataFolderCell{:}]';

%% Overwrite the Results to the File

save(ABS_PATH_TO_PATH_LOSSES_FILE, ...
    'pathLossesWithGpsInfo', 'relPathsOutFilesUnderDataFolder',...
    'maxMeasurablePathLossInfo');

disp('    Done!')
% EOF