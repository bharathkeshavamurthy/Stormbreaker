% COMPUTEOPTIANGLESFORAIRTOGROUNDMEASCAMP Simulate a few RX routes with the
% TX fixed on the ground to compute the optimal elevation angle for each
% scenario.
%
% Yaguang Zhang, Purdue, 03/13/2018

clear; clc; close all; dbstop if error;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'OptimalAngleForAirToGroundMeas');
summaryCsvFileName = 'SummaryForSimulationRestuls.csv';

% Planned routes. Each row is for one simulation.
txLatLonAlts = [ ...
    39.994562, -105.262983, 0; ... Sim 0 (A test)
    ...
    39.994562, -105.262983, 6.5; ... Sim 1 (Height: Building + cart)
    39.994562, -105.262983, 6.5; ... Sim 2
    39.994562, -105.262983, 6.5; ... Sim 3
    ...
    39.994562, -105.262983, 6.5; ... Sim 4
    39.994562, -105.262983, 6.5; ... Sim 5
    39.994562, -105.262983, 6.5; ... Sim 6
    ...
    39.992321, -105.273496, 1.5; ... Sim 7 (Height: cart)
    39.992321, -105.273496, 1.5; ... Sim 8
    39.992321, -105.273496, 1.5; ... Sim 9
    ...
    39.989103, -105.277764, 1.5 + 45; ... Sim 10 (Height: cart + elev gain)
    39.989103, -105.277764, 1.5 + 45; ... Sim 11 (Height: cart + elev gain)
    39.989103, -105.277764, 1.5 + 45; ... Sim 12 (Height: cart + elev gain)
    ...
    39.992321, -105.273496, 1.5; ... Sim 13 (Height: cart)
    39.992321, -105.273496, 1.5; ... Sim 14
    39.992321, -105.273496, 1.5; ... Sim 15
    ...
    39.989103, -105.277764, 1.5 + 45; ... Sim 16 (Height: cart + elev gain)
    39.989103, -105.277764, 1.5 + 45; ... Sim 17 (Height: cart + elev gain)
    39.989103, -105.277764, 1.5 + 45; ... Sim 18 (Height: cart + elev gain)
    ];
rxStartLatLonAlts = [ ...
    39.994562, -105.262983, 0; ... Sim 0
    ...
    39.994434, -105.263091, 30.48; ... Sim 1 (100 feet)
    39.994434, -105.263091, 60.96; ... Sim 2 (200 feet)
    39.994434, -105.263091, 91.44; ... Sim 3 (300 feet)
    ...
    39.993228, -105.262845, 30.48; ... Sim 4
    39.993228, -105.262845, 60.96; ... Sim 5
    39.993228, -105.262845, 91.44; ... Sim 6
    ...
    39.992018, -105.273909, 30.48; ... Sim 7
    39.992018, -105.273909, 60.96; ... Sim 8
    39.992018, -105.273909, 91.44; ... Sim 9
    ...
    39.989216, -105.277614, 30.48 + 45; ... Sim 10 (Height: + elev gain)
    39.989216, -105.277614, 60.96 + 45; ... Sim 11
    39.989216, -105.277614, 91.44 + 45; ... Sim 12
    ...
    39.991645, -105.275549, 30.48 + 15; ... Sim 13 (Height: + elev gain)
    39.991645, -105.275549, 60.96 + 15; ... Sim 14
    39.991645, -105.275549, 91.44 + 15; ... Sim 15
    ...
    39.991645, -105.275549, 30.48 + 15; ... Sim 16 (Height: + elev gain)
    39.991645, -105.275549, 60.96 + 15; ... Sim 17
    39.991645, -105.275549, 91.44 + 15; ... Sim 18
    ];
rxEndLatLonAlts = [ ...
    39.992384, -105.263995, 0; ... Sim 0
    ...
    39.992384, -105.263995, 30.48; ... Sim 1
    39.992384, -105.263995, 60.96; ... Sim 2
    39.992384, -105.263995, 91.44; ... Sim 3
    ...
    39.994113, -105.264598, 30.48; ... Sim 4
    39.994113, -105.264598, 60.96; ... Sim 5
    39.994113, -105.264598, 91.44; ... Sim 6
    ...
    39.989216, -105.277614, 30.48 + 45; ... Sim 7 (Height: + elev gain)
    39.989216, -105.277614, 60.96 + 45; ... Sim 8
    39.989216, -105.277614, 91.44 + 45; ... Sim 9
    ...
    39.992018, -105.273909, 30.48; ... Sim 10
    39.992018, -105.273909, 60.96; ... Sim 11
    39.992018, -105.273909, 91.44; ... Sim 12
    ...
    39.990528, -105.274463, 30.48 + 15; ... Sim 13 (Height: + elev gain)
    39.990528, -105.274463, 60.96 + 15; ... Sim 14
    39.990528, -105.274463, 91.44 + 15; ... Sim 15
    ...
    39.990528, -105.274463, 30.48 + 15; ... Sim 16 (Height: + elev gain)
    39.990528, -105.274463, 60.96 + 15; ... Sim 17
    39.990528, -105.274463, 91.44 + 15; ... Sim 18
    ];

maxMisalignmentAngleAllowed = 10;

% Simulation info.
simInfo = { ...
    'Sim-0_DebugTest'; ... Sim 0
    ...
    'Sim-1_TestSite-1a_Building-1_100ft'; ... Sim 1
    'Sim-2_TestSite-1a_Building-1_200ft'; ... Sim 2
    'Sim-3_TestSite-1a_Building-1_300ft'; ... Sim 3
    ...
    'Sim-4_TestSite-1b_Building-1_100ft'; ... Sim 4
    'Sim-5_TestSite-1b_Building-1_200ft'; ... Sim 5
    'Sim-6_TestSite-1b_Building-1_300ft'; ... Sim 6
    ...
    'Sim-7_TestSite-2a_TX-1_100ft'; ... Sim 7
    'Sim-8_TestSite-2a_TX-1_200ft'; ... Sim 8
    'Sim-9_TestSite-2a_TX-1_300ft'; ... Sim 9
    ...
    'Sim-10_TestSite-2a_TX-2_100ft'; ... Sim 10
    'Sim-11_TestSite-2a_TX-2_200ft'; ... Sim 11
    'Sim-12_TestSite-2a_TX-2_300ft'; ... Sim 12
        ...
    'Sim-13_TestSite-2b_TX-1_100ft'; ... Sim 13
    'Sim-14_TestSite-2b_TX-1_200ft'; ... Sim 14
    'Sim-15_TestSite-2b_TX-1_300ft'; ... Sim 15
        ...
    'Sim-16_TestSite-2b_TX-2_100ft'; ... Sim 16
    'Sim-17_TestSite-2b_TX-2_200ft'; ... Sim 17
    'Sim-18_TestSite-2b_TX-2_300ft'; ... Sim 18
    };

%% Simulations
disp(' ----------------------------------------- ')
disp('  computeOptiAnglesForAirToGroundMeasCamp ')
disp(' ----------------------------------------- ')

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

summaryFileID ...
    = fopen(fullfile(ABS_PATH_TO_SAVE_PLOTS, summaryCsvFileName), 'w');
fprintf(summaryFileID, '%s\n', ...
    'SimInfo, optiEleAngleTxInDeg, usedAziAngleTxInDeg');

[numSims, ~] = size(txLatLonAlts);
for idxSim = 1:numSims
    disp(['    Simulation ', num2str(idxSim), '/', ...
        num2str(numSims), '...']);
    [optiEleAngleTx, usedAziAngleTx,~] ...
        = simulateAirToGroundMeas( txLatLonAlts(idxSim, :), ...
        rxStartLatLonAlts(idxSim, :), rxEndLatLonAlts(idxSim, :), ...
        maxMisalignmentAngleAllowed, ...
        fullfile(ABS_PATH_TO_SAVE_PLOTS, [simInfo{idxSim}, '_']));
    
    % Record the result into a txt file.
    fprintf(summaryFileID, '%s,%.2f,%.2f\n', ...
        simInfo{idxSim}, optiEleAngleTx, usedAziAngleTx);
end

fclose(summaryFileID);
disp('    Done!')

% EOF