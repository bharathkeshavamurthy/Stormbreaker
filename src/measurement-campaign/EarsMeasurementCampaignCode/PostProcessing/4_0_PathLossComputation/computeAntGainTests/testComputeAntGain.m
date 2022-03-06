% TESTCOMPUTEANTGAIN This script will run the function computeAntGain with
% some dummy inputs to make sure the (az, el) transform inside it works as
% expected.
%
% Yaguang Zhang, Purdue, 10/05/2017

clear; clc; close all; dbstop if error;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd(fullfile('..', '..')); setPath;

% We also need the functions antPatInter.m and computeAntGain.m for antenna
% gain calculation.
addpath(fullfile(pwd, '3_AntennaPattern'));
addpath(fullfile(pwd, '4_0_PathLossComputation'));

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'ComputeAntGainTests');

% Reuse results from fetchAntennaPattern.m and loadMeasCampaignInfo.m.
ABS_PATH_TO_ANT_PAT_FILE = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'AntennaPattern', 'antennaPattern.mat');
ABS_PATH_TO_TX_INFO_LOGS_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');

%% Before Processing the Data

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

assert(exist(ABS_PATH_TO_ANT_PAT_FILE, 'file')==2, ...
    'Couldn''t find antennaPattern.mat! Please run PostProcessing/3_AntennaPattern/fetchAntennaPattern.m first.');
assert(exist(ABS_PATH_TO_TX_INFO_LOGS_FILE, 'file')==2, ...
    'Couldn''t find txInfoLogs.mat! Please run PostProcessing/4_0_PathLossComputation/loadMeasCampaignInfo.m first.');

% Get 'pat28GAzNorm', and 'pat28GElNorm'.
load(ABS_PATH_TO_ANT_PAT_FILE);
% Get records of the TxInfo.txt files (among other contant parameters for
% the measurement campaign, e.g. F_S, TX_LAT, TX_LON, and TX_POWER_DBM):
% 'TX_INFO_LOGS' and 'TX_INFO_LOGS_ABS_PAR_DIRS'.
load(ABS_PATH_TO_TX_INFO_LOGS_FILE);

%% Dummy Data to Use

% We need:
%    lat0, lon0, alt0, ...
%     az0, el0, ...
%    lat, lon, alt, ...
%     antPatAz, antPatEl, FLAG_DEBUG
% We will use the TX GPS position on the tower to fill the origin location
% parameters and ultilize the antenna pattern we got for our horn-shape
% antennas.

lat0 = TX_LAT;
lon0 = TX_LON;
alt0 = 100;     % For simplicity.
az0 = [45 50 135 140 45  50   135 140 -40 -50 -110 -125 -40 -50 -110 -125];
el0 = [45 -30 50 -45 120 -115 160 -155 120 -115 160 -155 45 -30 50 -45];
lats = lat0 + [-0.001 -0.001 0.001 0.001 -0.001 -0.001 0.001 0.001];
lons = lon0 + [-0.001 0.001 -0.001 0.001 -0.001 0.001 -0.001 0.001];
alts = alt0 + [50 50 50 50 -50 -50 -50 -50];

%% Tests
for idxAng = 1:length(az0)
    for idxCoor = 1:length(lats)
        [~, hDebugFig, hDebugFigMap] = computeAntGain(lat0, lon0, alt0, ...
            az0(idxAng), el0(idxAng), ...
            lats(idxCoor), lons(idxCoor), alts(idxCoor), ...
            pat28GAzNorm, pat28GElNorm, true);
        
        % Save the plot.
        curAbsPathToSavePlot = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
            ['testComputeAntGainFig_', num2str(idxAng), '_', ...
            num2str(idxCoor)]);
        saveas(hDebugFig, [curAbsPathToSavePlot, '.png']);
        saveas(hDebugFig, [curAbsPathToSavePlot, '.fig']);
        
        saveas(hDebugFigMap, [curAbsPathToSavePlot, '_map.png']);
        saveas(hDebugFigMap, [curAbsPathToSavePlot, '_map.fig']);
    end
    
    % To save memory.
    close all;
end

% EOF