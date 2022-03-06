% COMPAREPATHLOSSES Compare the original path losses with the ones computed
% using shifted data (e.g. shifted antenna angles) to test the sensitivity
% of the path loss computation procedure.
%
% We have evaluated the path losses for SIMO & Large-scale sites with extra
% artificial measurement errors and copied the results under the parent
% directory of this script for generating plots.
%
% Yaguang Zhang, Purdue, 02/14/2018

clear; clc; close all; dbstop if error;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Path to save plots.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputationWithUniformAntAngleOffsets');

% Orignal.
ABS_PATH_TO_PATH_LOSSES_ORI = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation');
ABS_PATH_TO_TX_INFO_LOGS_ORI = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');
% Path losses via uniformly shifted antenna angles.
ABS_PATH_TO_NEW_PATH_LOSSES = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'EarsMeasurementCampaignCode', 'PostProcessing', ...
    '9_1_PathLossComputationWithAntAngleOffsets', ...
    'NewEvalPathLossesResults');
ABS_PATH_TO_PATH_LOSSES_UNI_1 = fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    'UniOffsetsOnAntAngles_-5_5_Az_-2_2_El');
ABS_PATH_TO_PATH_LOSSES_UNI_2 = fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    'UniOffsetsOnAntAngles_-5_5_Az');

%% Load Path Loss Computation Results

% Orignal.
pathLossesOri = load(fullfile(ABS_PATH_TO_PATH_LOSSES_ORI, 'pathLossesWithGpsInfo.mat'));
pathLossesWithGpsInfoOri = pathLossesOri.pathLossesWithGpsInfo;

txInfoOri = load(ABS_PATH_TO_TX_INFO_LOGS_ORI);
txInfoLogsOri = txInfoOri.TX_INFO_LOGS;

% Path losses via uniformly shifted antenna angles.
pathLossesUni_1 = load(fullfile(ABS_PATH_TO_PATH_LOSSES_UNI_1, 'pathLossesWithGpsInfo.mat'));
pathLossesWithGpsInfoUni_1 = pathLossesUni_1.pathLossesWithGpsInfo;
txInfoLogsUni_1 = pathLossesUni_1.txInfoLogsNew;

pathLossesUni_2 = load(fullfile(ABS_PATH_TO_PATH_LOSSES_UNI_2, 'pathLossesWithGpsInfo.mat'));
pathLossesWithGpsInfoUni_2 = pathLossesUni_2.pathLossesWithGpsInfo;
txInfoLogsUni_2 = pathLossesUni_2.txInfoLogsNew;

%% Comparisons

% Comparison 1.
[hFigPathLossCompUni1, hFigCompDifferences1] ...
    = compareTwoSetsOfPathLosses(pathLossesWithGpsInfoOri, ...
    pathLossesWithGpsInfoUni_1);
set(0, 'CurrentFigure', hFigPathLossCompUni1)
title({'Path Losses - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and [-2,2] degrees on El.)'});
set(0, 'CurrentFigure', hFigCompDifferences1)
title({'Path Loss Differences - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and [-2,2] degrees on El.)'});

hFigAntAngleCompUni1 = compareTwoSetsOfAntAngles(txInfoLogsOri, txInfoLogsUni_1);
title({'Antenna Angles - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and [-2,2] degrees on El.)'});

filenamePrefix = 'UniAntAng_-5_5_Az_-2_2_El_';
saveas(hFigPathLossCompUni1, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'PathLossComp', '.png']));
saveas(hFigCompDifferences1, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'PathLossDiff', '.png']));
saveas(hFigAntAngleCompUni1, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'AngleDiff', '.png']));

% Comparison 2.
[hFigPathLossCompUni2, hFigCompDifferences2] ...
    = compareTwoSetsOfPathLosses(pathLossesWithGpsInfoOri, ...
    pathLossesWithGpsInfoUni_2);
set(0, 'CurrentFigure', hFigPathLossCompUni2)
title({'Path Losses - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and no offset on El.)'});
set(0, 'CurrentFigure', hFigCompDifferences2)
title({'Path Loss Differences - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and no offse on El.)'});

hFigAntAngleCompUni2 = compareTwoSetsOfAntAngles(txInfoLogsOri, txInfoLogsUni_2);
title({'Antenna Angles - Uniform Offsets on Antenna Angles', ...
    '([-5, 5] degrees on Az. and no offse on El.)'});

filenamePrefix = 'UniAntAng_-5_5_Az_';
saveas(hFigPathLossCompUni2, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'PathLossComp', '.png']));
saveas(hFigCompDifferences2, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'PathLossDiff', '.png']));
saveas(hFigAntAngleCompUni2, fullfile(ABS_PATH_TO_NEW_PATH_LOSSES, ...
    [filenamePrefix, 'AngleDiff', '.png']));

% EOF