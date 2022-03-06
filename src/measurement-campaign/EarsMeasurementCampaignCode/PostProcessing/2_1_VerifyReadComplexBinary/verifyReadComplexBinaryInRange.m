% VERIFYREADANDWRITECOMPLEXBINARYINRANGE Verify that the functions
% countComplexBinary and readComplexBinaryInRange are working as expected.
%
% Yaguang Zhang, Purdue, 10/06/2017

clear; clc; close all;

%% Configuration

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
curPath = pwd;
addpath(fullfile(pwd));
cd('..'); setPath;

% Files to read from and write into.
refFile = which('measureSignal_1497560832_filtered.out');
genFile = fullfile(curPath, 'genFile.out');

RANGE = [100, 200];

%% Load Data
refSig = read_complex_binary(refFile);

numSigSamps= countComplexBinary(refFile);
sigInRange = readComplexBinaryInRange(refFile, RANGE);

%% Compare the Results
assert(length(refSig) == numSigSamps, 'Error: countComplexBinary does not work!');
assert(all( ...
    refSig(RANGE(1):RANGE(2))-sigInRange==0 ...
    ), 'Error: readComplexBinaryInRange does not work!');

disp('Tests passed!')

% EOF