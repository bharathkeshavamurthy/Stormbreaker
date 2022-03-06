function [ hInterPat3D ] = plotInterPat3D( patAz, patEl, ...
    INTER_METHOD, FLAG_INTER_IN_DB, numPtsPerDim, ...
    FLAG_USE_MATLAB_AZEL_CONVENTION)
%PLOTINTERPAT3D Plot in 3D the antenna pattern interpolation results for a
%specified method.
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
%   - INTER_METHOD
%     A string for the interpolation method to use. This should be
%     supported by antPatInter.m.
%   - FLAG_INTER_IN_DB
%     Set this to be true if the interpolation should be carried out using
%     the amplitude in dB (instead of using the raw linear amplitudes).
%   - numPtsPerDim
%     A scalar specifying how many points should be generated for each
%     angle dimension.
% Output:
%   - hInterPat3D
%     The output figure handler.
%
% Yaguang Zhang, Purdue, 10/03/2017

if nargin < 4
    % By default, we will interpolate directly using the linear amplitudes.
    FLAG_INTER_IN_DB = false;
end
if nargin < 5
    % For illustration, by default we will use 100 points for each axis.
    numPtsPerDim = 100;
end
if nargin < 6
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

[ x,y,z,ampsDb, ~, X,Y,Z,~ ] = interpAntPatsIn3D( ...
    patAz, patEl, ...
    numPtsPerDim, ...
    FLAG_INTER_IN_DB, INTER_METHOD, FLAG_USE_MATLAB_AZEL_CONVENTION);

hInterPat3D = figure('units','normalized', ...
    'outerposition',[0.1 0.05 0.8 0.9]);
colormap jet; hold on;
plot3k([x(:),y(:),z(:)], 'ColorData', ampsDb(:));
hRef = plot3(X,Y,Z, '.-.', 'Color', ones(1,3).*0.7);
hold off;
if FLAG_USE_MATLAB_AZEL_CONVENTION
    xlabel('x (to front)');
    ylabel('y (to antenna''s left-hand side)');
    zlabel('z (to top)');
else
    xlabel('x (to antenna''s right-hand side)');
    ylabel('y (to front)');
    zlabel('z (to top)');
end
title({'Interpolated Antenna 3D Radiation Pattern'; ...
    '(Amplitude in dB)'});
legend(hRef,'Ref', 'Location', 'southeast');
axis equal; view(135,30);
grid on;

end
% EOF