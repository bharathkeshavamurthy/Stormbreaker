% VERIFYREADANDWRITECOMPLEXBINARY Verify that the functions
% read_complex_binary and write_complex_binary are working as expected.
%
% Yaguang Zhang, Purdue, 09/13/2017

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

%% Read and Write
refSig = read_complex_binary(refFile);
write_complex_binary(refSig, genFile);

%% Read in genFile and Compare the Results
genSig = read_complex_binary(genFile);
assert(all(refSig-genSig==0), 'The signal in genFile.out should be exactly the same as that read from the original file.');

disp('Test passed: ')
disp('    The signal in genFile.out is indeed the same as that read from the original file.')
disp(' ')
disp('Please run compareTwoOutFiles.grc in Gnu Radio to compare the signals.')
% EOF