% POSTPROCESSMEASDATA Post process the measurement dataset.
%
% This is a holder script listing all the steps required to properly
% process the measurement dataset. Please comment & uncomment commands as
% it is needed, depending on which results require updates.
%
% Yaguang Zhang, Purdue, 10/09/2017

clear; clc; close all;

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(pwd)));
setPath;

%% 1_SummaryReport: First run of the dataset.
% This will generate: 'allSeriesParentDirs', 'allSeriesDirs'.

genPlots;
genLatexForPlots;

%% 2_0_Calibration: Calibrate the Gnu Radio RX.
% This will generate: 'lsLinesPolys', 'lsLinesPolysInv', 'fittedMeaPs',
% 'fittedCalPs', 'rxGains'.

calibrateRx;
plotCalibrationLines;

%% 3_AntennaPattern: Generate the Antenna Pattern
% This will generate: 'pat28GAzNorm', 'pat28GElNorm'.

fetchAntennaPattern;

%% 4_0_PathLossComputation: Compute & Plot the Pass Losses for Each Site
% This will generate: 'TX_POWER_DBM', 'TX_HEIGHT_FEET', 'TX_HEIGHT_M',
% 'F_S', 'TX_LAT', 'TX_LON', 'TX_INFO_LOGS', 'TX_INFO_LOGS_ABS_PAR_DIRS'.

loadMeasCampaignInfo;

evalPathLosses;
evalPathLossesForContiTracks;

%% 4_1_PlotPathLossesByCategory: Plot the Path Losses by Category

plotBasicTransLossesByCategory;

%% 5_ComparePathLossesWithItuModels: Compute Ref ITU Results

compareSiteGenOverRoofTopsLoS;
compareSiteGenOverRoofTopsNLoS;

%% 6_MapPathLossesToPlusPatterns: Generate Path Loss over Possition Plots
% The pattern will look like a "+" sign.
mapPathLossesToPlusPatterns;

%% 7_SimpleModelsForNLoSByBuildings: Modeling Building Blockage
simpleModelsForNLoSByBuildings;

%% 8_SimpleModelsForNLoSByVegetations: Modeling Foliage Blockage
simpleModelsForNLoSByVegetations;

% EOF