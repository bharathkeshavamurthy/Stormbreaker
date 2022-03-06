function [ properAzDiff ] = fixAzDiffForPlotting( azDiff )
%FIXAZDIFFFORPLOTTING Move the input angle to [-180, 180) if it is not
%already within that range.
%
% Yaguang Zhang, Purdue, 02/22/2018

if azDiff>=180
    newAzDiff = azDiff-360;
elseif azDiff<-180
    newAzDiff = azDiff+360;
else
    newAzDiff = azDiff;
end
properAzDiff = newAzDiff;

if azDiff~=newAzDiff
      disp(['    Angle difference ', num2str(azDiff), ...
          ' is out of [-180, 180)... Changed it to ', num2str(newAzDiff)]);
end

end

