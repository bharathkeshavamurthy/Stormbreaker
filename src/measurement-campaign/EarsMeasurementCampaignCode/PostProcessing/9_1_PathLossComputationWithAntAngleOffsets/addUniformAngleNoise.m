function [ anglesWithUniformNoise ] = addUniformAngleNoise( angles, noiseMin, noiseMax )
%ADDUNIFORMANGLENOISE Add to the input column vector angles (in degree) a
%uniform noise vector with each element independently sampled in (noiseMin,
%noiseMax).
%
% Yaguang Zhang, Purdue, 02/13/2018

anglesWithUniformNoise ...
    = angles + noiseMin + (noiseMax-noiseMin)*rand(length(angles),1);

end

% EOF