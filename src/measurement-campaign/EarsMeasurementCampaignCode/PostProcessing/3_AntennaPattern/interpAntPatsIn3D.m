function [ x,y,z,ampsDb,ampsDbRel, X,Y,Z,AMPSDB,AMPSDBREL ] ...
    = interpAntPatsIn3D( ...
    patAz, patEl, ...
    numPtsPerDim, ...
    FLAG_INTER_IN_DB, INTER_METHOD, FLAG_USE_MATLAB_AZEL_CONVENTION)
%INTERPANTPATSIN3D Interperlates the antenna patterns for azimuth and
%elevation planes in 3D for plotting.
%
% Inputs:
%   - patAz, patEl
%     The reference antenna patterns, for the Azimuth and the Elevation
%     sweep, respectively; Each of them is a struct containing fields:
%       - azs
%         The azimuth angles in degree.
%       - els
%         The elevation angles in degree.
%       - amps
%         The linear amplitudes of the samples.
%       - phases
%         The phases of the samples.
%     All of these fields contains a column vector with each row
%     corresponding to a sweep sample.
%   - numPtsPerDim
%     For calculating the grid step for each angle axis in the range of [0,
%     360) degrees. It is essentially the number of needed interpolation
%     points needed for the azimuth (and the elevation) axis.
%   - FLAG_INTER_IN_DB
%     Set this to be true to linearly interpolate the dB version of the
%     antenna patterns (an extra antPatLinearToDb.m step will be preformed
%     before the interpolation); Otherwise, the interpolcation will be
%     carried out as the original input patterns are.
%   - INTER_METHOD
%     The interpolation method used for antPatInter.m. 'WeightedSum' is
%     recommended.
%   - FLAG_USE_MATLAB_AZEL_CONVENTION
%     Set this to be true (recommended) to use Matlab built-in function
%     sph2cart for coordinate system tranformation.
%
% Outputs:
%   - x,y,z,ampsDb,ampsDbRel
%     Matrices for the 3D interpolated pattern. x, y and z are the
%     coordinates for the antenna pattern points, and ampsDb has the
%     corresponding antenna gains for them. And 
%         ampsDbRel = ampsDb-min(ampsDb).
%   - X,Y,Z,AMPSDB
%     Similarly, matrices for the reference patterns in 3D.
%
% Yaguang Zhang, Purdue, 10/16/2017

% We will only need [0, 360) for elevation and [0, 180) for azimuth to
% cover the whole space.
azs = linspace(0,180,ceil(numPtsPerDim/2)+1);
els = linspace(0,360,numPtsPerDim+1);
% Discard the last point which corresponds to 360 degrees but is
% essentially 0.
azs = azs(1:(end-1));
els = els(1:(end-1));
[azs, els] = meshgrid(azs, els);

% We need to carry out an interpolation accordingly to the antenna S21
% measurement results.
if FLAG_INTER_IN_DB
    patAzDb = patAz;
    patAzDb.amps = antPatLinearToDb(patAzDb.amps);
    patElDb = patEl;
    patElDb.amps = antPatLinearToDb(patElDb.amps);
    ampsDb = antPatInter(patAzDb, patElDb, azs, els, INTER_METHOD);
else
    amps = antPatInter(patAz, patEl, azs, els, INTER_METHOD);
    % Plot in dB.
    ampsDb = antPatLinearToDb(amps);
end

% We will also plot the sweep data, just for reference.
AZS = [patAz.azs; zeros(length(patEl.azs),1)];
ELS = [zeros(length(patAz.els),1); patEl.els];
AMPS = [patAz.amps; patEl.amps];
AMPSDB = antPatLinearToDb(AMPS);

% Shift all the amplitudes in dB to nonegative values.
minAmpDb = min([ampsDb(:);AMPSDB(:)]);
ampsDbRel = ampsDb - minAmpDb;
AMPSDBREL = AMPSDB - minAmpDb;

if FLAG_USE_MATLAB_AZEL_CONVENTION
    [x,y,z] = sph2cart(deg2rad(azs),deg2rad(els),ampsDbRel);
    [X,Y,Z] = sph2cart(deg2rad(AZS),deg2rad(ELS),AMPSDBREL);
else
    % Convert from the polar coordinate system to the Cartesian system for
    % plotting. We have
    %    x   = amp * cosd(el) * sind(360-az)
    %     y  = amp * cosd(el) * cosd(360-az)
    %      z = amp * sind(el)
    x = ampsDbRel .* cosd(els) .* sind(360-azs);
    y = ampsDbRel .* cosd(els) .* cosd(360-azs);
    z = ampsDbRel .* sind(els);
    
    X = AMPSDBREL .* cosd(ELS) .* sind(360-AZS);
    Y = AMPSDBREL .* cosd(ELS) .* cosd(360-AZS);
    Z = AMPSDBREL .* sind(ELS);
end
end
% EOF