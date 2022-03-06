function [ hPat3DRef ] = plotRefPat3D( patAz, patEl, ...
    FLAG_USE_MATLAB_AZEL_CONVENTION)
%PLOTREFPAT3D Plot in 3D the reference antenna pattern.
%
% Inputs:
%   - patAz, patEl
%     The antenna patterns, for the Azimuth and Elevation sweeps,
%     respectively; Each of which is a struct containing fields:
%       - azs
%         The azimuth angles in degree from set [0, 360).
%       - els
%         The elevation angles in degree from set [0, 360).
%       - amps
%         The linear amplitudes of the samples.
%       - phases
%         The phases of the samples.
%     All of these fields contains a column vector with each row
%     corresponding to a sweep sample.
% Output:
%   - hPat3DRef
%     The output figure handler.
%
% Yaguang Zhang, Purdue, 10/03/2017

if nargin<3
    % The Matlab convention about Azimuth and Elevation:
    %
    %   The azimuth angle of a vector is the angle between the x-axis and
    %   the orthogonal projection of the vector onto the xy plane. The
    %   angle is positive in going from the x axis toward the y axis.
    %   Azimuth angles lie between –180 and 180 degrees. The elevation
    %   angle is the angle between the vector and its orthogonal projection
    %   onto the xy-plane. The angle is positive when going toward the
    %   positive z-axis from the xy plane. These definitions assume the
    %   boresight direction is the positive x-axis.
    %
    FLAG_USE_MATLAB_AZEL_CONVENTION = true;
end

AZS = [patAz.azs; zeros(length(patEl.azs),1)];
ELS = [zeros(length(patAz.els),1); patEl.els];
AMPS = [patAz.amps; patEl.amps];
AMPSDB = antPatLinearToDb(AMPS);

% Shift all the amplitudes in dB to nonegative values.
minAmpDb = min(AMPSDB(:));
AMPSDBREL = AMPSDB - minAmpDb;

if FLAG_USE_MATLAB_AZEL_CONVENTION
    [X,Y,Z] = sph2cart(deg2rad(AZS),deg2rad(ELS),AMPSDBREL);
else
    % Convert from the polar coordinate system to the Cartesian system for
    % plotting. We have
    %    x   = amp * cosd(el) * sind(360-az)
    %     y  = amp * cosd(el) * cosd(360-az)
    %      z = amp * sind(el)
    X = AMPSDBREL .* cosd(ELS) .* sind(360-AZS);
    Y = AMPSDBREL .* cosd(ELS) .* cosd(360-AZS);
    Z = AMPSDBREL .* sind(ELS);
end

hPat3DRef = figure('units','normalized', ...
    'outerposition',[0.1 0.05 0.8 0.9]);
colormap jet;
plot3k([X,Y,Z], 'ColorData', AMPSDB);
if FLAG_USE_MATLAB_AZEL_CONVENTION
    xlabel('x (to front)');
    ylabel('y (to antenna''s left-hand side)');
    zlabel('z (to top)');
else
    xlabel('x (to antenna''s right-hand side)');
    ylabel('y (to front)');
    zlabel('z (to top)');
end
title({'Reference Antenna 3D Radiation Pattern'; ...
    '(Amplitude in dB)'});
axis equal; view(135,30);
grid on;

end
% EOF