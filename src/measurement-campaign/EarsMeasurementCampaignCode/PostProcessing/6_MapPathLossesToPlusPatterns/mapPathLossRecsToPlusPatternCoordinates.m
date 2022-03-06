function [ xs, zs, hMissingSamps ] ...
    = mapPathLossRecsToPlusPatternCoordinates(relPaths, fInHz)
%MAPPATHLOSSRECSTOPLUSPATTERNCOORDINATES Map the measurements back to the
%plus pattern on which these measurements were carried out.
%
% Input:
%    - relPaths
%      The relative paths for the .out files to be mapped. Note that they
%      have to be all the measurements for one site.
%    - fInHz
%      Signal frequency. Default 28 GHz = 28 000 000 000 Hz.
%
% Output:
%    - xs, zx
%      Column vectors specifying the coordinates on the plus pattern. We
%      will have a right-hand coordinate system of (x, z) in meter where
%      the RX antenna points at +y. More specificially, +x is the
%      right-hand side of the antenna and +z is the top of it. (0, 0) is
%      the home position (x at center, z at top) of the platform.
%
% Yaguang Zhang, Purdue, 10/16/2017

if nargin<2
    fInHz = 28.*10.^9;
end

% Wavelength.
lambdaInM = physconst('LightSpeed')./(fInHz);

% Check the validity of the input.
if isempty(relPaths)
    error('Inpute relPaths should be non-empty!');
end
[ dates, types, serNums, timestamps ] ...
    = cellfun(@(p) parseOutFileRelPath(p), relPaths, ...
    'UniformOutput', false);
assert(all([length(unique(dates)), length(unique(types)), ...
    all([serNums{:}] == serNums{1}), all(diff([timestamps{:}])~=0)]), ...
    'Inpute relPaths should be for unique .out files in the same parent directory!');

% Debug plot.
hMissingSamps = nan;
switch types{1}
    % For LargeScale and SIMO, 2e will only consider two cases for each
    % site: (1) Only 1/2 measurement(s) missing; (2) Half of the
    % measurements are missing.
    case 'LargeScale'
        % Start at home (0,0): step lambda -> 19 -z -> 10 +z 10 -x -> 20 +x
        step = lambdaInM;
        numSampsPerAxis = 20;
        [xs, zs, hMissingSamps] ...
            = assignPlusPatXZCoors(step, numSampsPerAxis, timestamps);
    case 'SIMO'
        % Start at home (0,0): step lambda/4 -> 159 -z -> 80 +z 80 -x ->
        % 160 +x
        step = lambdaInM/4;
        numSampsPerAxis = 160;
        [xs, zs, hMissingSamps] ...
            = assignPlusPatXZCoors(step, numSampsPerAxis, timestamps);
    case 'Conti'
        %
        [xs, zs] = deal(zeros(length(relPaths),1));
    otherwise
        error(['Unsupported measurement type ', types]);
end

end
% EOF