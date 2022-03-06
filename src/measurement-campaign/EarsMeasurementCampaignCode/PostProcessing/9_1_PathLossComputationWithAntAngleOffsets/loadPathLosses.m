function [ pathLossesWithGpsInfo, txInfoLogs] = loadPathLosses( absPathToDir )
%LOADPATHLOSSES Load the path losses and TX info logs in a directory.
%
% Yaguang Zhang, Purdue, 02/14/2018

pathPathLossFileToSave = fullfile(absPathToDir, ...
    'pathLossesWithGpsInfo.mat');

loadedVars = load(pathPathLossFileToSave);



end
% EOF