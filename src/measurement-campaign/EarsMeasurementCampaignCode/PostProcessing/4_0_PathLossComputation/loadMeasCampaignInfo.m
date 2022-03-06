% LOADMEASCAMPAIGNINFO Load the measurement campaign records from
% TxInfo.txt.
%
%   When available, parameters including series number, location, TxAz,
%   TxEl, RxAz, TxEl and note, will be loaded. We will also hardcode some
%   constant parameters here.
%
%   We will use all capitalized variable names for the parameters we
%   generate here.
%
% Yaguang Zhang, Purdue, 10/04/2017

clear; clc; close all;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_RESULTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation');

%% Hard Coded Parameters

% Tx Power in dBm.
TX_POWER_DBM = 23;
% Tx tower height in feet.
TX_HEIGHT_FEET = 90;

% Sample rate used for GnuRadio.
F_S = 1.04 * 10^6;

% Signal frequency in GHz.
F_C_IN_GHZ = 28;

% Transmitter location.
TX_LAT = 38.983899;
TX_LON = -76.486682;

% The downconverter gain at the RX side.
DOWNCONVERTER_GAIN_IN_DB = 13.4; 

%% Necessary Unit Conversion
TX_HEIGHT_M = distdim(TX_HEIGHT_FEET,'feet','meters');

%% Load Records from TxInfo.txt Files

% Find all TxInfo.txt log files.
pathToData = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, 'Data');
txInfoFilesDirs = rdir(fullfile(pathToData, '**', 'TxInfo.txt'), '', false);

% We will save the result in a column cell, corresponding to a parent dir,
% with each item being a struct array for all the measurement series under
% that folder.
[TX_INFO_LOGS, TX_INFO_LOGS_ABS_PAR_DIRS] ...
    = arrayfun(@(logDir) parseTxInfoLog(logDir.name), ...
    txInfoFilesDirs, 'UniformOutput', false);

%% Save the Results
pathFileToSaveResults = fullfile(ABS_PATH_TO_SAVE_RESULTS, ...
    'txInfoLogs.mat');
save(pathFileToSaveResults, 'TX_POWER_DBM', ...
    'TX_HEIGHT_FEET', 'TX_HEIGHT_M', ...
    'F_S', 'F_C_IN_GHZ', 'TX_LAT', 'TX_LON', ...
    'TX_INFO_LOGS', 'TX_INFO_LOGS_ABS_PAR_DIRS', ...
    'DOWNCONVERTER_GAIN_IN_DB');
% EOF