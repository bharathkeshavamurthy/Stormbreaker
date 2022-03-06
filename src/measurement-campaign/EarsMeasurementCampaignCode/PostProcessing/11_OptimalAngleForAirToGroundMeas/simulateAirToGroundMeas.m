function [ optiEleAngleTx, usedAziAngleTx, hSimuFigs ] ...
    = simulateAirToGroundMeas( txLatLonAlt, ...
    rxStartLatLonAlt, rxEndLatLonAlt, maxMisalignmentAngleAllowed, ...
    pathToSaveFig )
%SIMULATEAIRTOGROUNDMEAS Simulate the air-to-ground measurement in 3D to
%get the optimal elevation angle for the TX & RX.
%
% Inputs:
%   - txLatLonAlt
%     [Latitude in degree, longitude in degree, altitude in meter] for the
%     TX.
%   - rxStartLatLonAlt, rxEndLatLonAlt
%     [Latitude in degree, longitude in degree, altitude in meter] for the
%     start point and end point, respectively, of the RX's planned route.
%   - maxMisalignmentAngleAllowed
%     The maximum angle in degree that is allowed for the misalignement of
%     the TX and RX beams. For example, this can be the 3dB half beamwith
%     angle so that we know what elevation angle to choose for a
%     measurement route to make the TX and RX align within that range for
%     the most of the route.
%   - pathToSaveFig
%     Optional. A string specifiying the absolute path for saving the
%     output figures. When it is explicitly speficied, the figures will be
%     saved accordingly; Otherwise, the figures will not be saved.
%
% Outputs:
%   - optiEleAngleTx
%     The optimal elevation angle gotten for the TX. It is optimal in the
%     sense that it will keep the drone within the max. misalignment angle
%     range for both the TX and the RX during the greatest portion of the
%     flight.
%   - usedAziAngleTx
%     The azimuth angle in [0, 360) degrees used for the TX during the
%     simulation.
%   - hSimuFigs
%     A vector cell containing the handlers for the simulation result
%     figures.
%
% More about the measurement scenario:
%   The TX will be mounted and fixed, facing to the middle point of the RX
%   route. The hSimuFigsRX will be moved on a drone with fixed elevation
%   and azimuth angles so that it is facing at the opposite direction of
%   that for the TX.
%
% Yaguang Zhang, Purdue, 03/12/2018

%% Settings
% Num of points to be simulated on one RX route.
numPtsOnRoute = 1000;
% The length in meter for the direction vectors shown in plots.
directionLenToRouteLenRatio = 0.1;
% The elevation angles in degree to be investigated for the RX.
txEleAngels = [-90:0.1:90];
% Error that is allowed by "approximately equal", which is used for
% comparing coordinate elements (in meter) of points in the (x, y, lat)
% system.
errorApproxEqual = 0.0001;

%% Preprocessing

approxEqual = @(u, v) u<=v+errorApproxEqual & u>=v-errorApproxEqual;

% Convert GPS locations to UTM coordinates.
[txX, txY, txUtmZone] = deg2utm(txLatLonAlt(1), txLatLonAlt(2));
[rxStartX, rxStartY, rxStartUtmZone] ...
    = deg2utm(rxStartLatLonAlt(1), rxStartLatLonAlt(2));
[rxEndX, rxEndY, rxEndUtmZone] ...
    = deg2utm(rxEndLatLonAlt(1), rxEndLatLonAlt(2));
assert( ...
    all(strcmp({txUtmZone, rxStartUtmZone, rxEndUtmZone}, txUtmZone)), ...
    'Not able to deal with location points of different UTM zones!');

% Construct the flight route in the (x, y, altitude) space.
rxXs = linspace(rxStartX, rxEndX, numPtsOnRoute)';
rxYs = linspace(rxStartY, rxEndY, numPtsOnRoute)';
rxAlts = linspace(rxStartLatLonAlt(3), rxEndLatLonAlt(3), numPtsOnRoute)';
% Convert them back to GPS locations for plotting.
[rxLats, rxLons] ...
    = utm2deg(rxXs, rxYs, repmat(txUtmZone, numPtsOnRoute, 1));
% Compute how long we will plot the direction on the figures.
directionLength = norm([rxXs(1), rxYs(1), rxAlts(1)] ...
    - [rxXs(end), rxYs(end), rxAlts(end)]).*directionLenToRouteLenRatio;
% The transmitter location.
txAlt = txLatLonAlt(3);

% Evaluate usedAziAngleTx.
degBetweenTwo3DVectors = @(p1, p2) ...
    rad2deg(atan2(norm(cross(p1,p2)),dot(p1,p2)));
rxMiddlePt = mean([rxStartX, rxEndX; ...
    rxStartY, rxEndY; ...
    rxStartLatLonAlt(3), rxEndLatLonAlt(3)], 2);
txDirectDefault = rxMiddlePt - [txX; txY; txAlt];
usedAziAngleTx = mod( ...
    degBetweenTwo3DVectors( [0, 1, 0], ...
    [txDirectDefault(1), txDirectDefault(2), 0])...
    .*sign(txDirectDefault(1)), ...
    360);
usedAziAngleTxRad = deg2rad(usedAziAngleTx);
[txDirYDefault, txDirXDefault, txDirAltDefault] ...
    = sph2cart(usedAziAngleTxRad, ...
    0, directionLength);
% Normalize the x and y components to check future TX directions.
lengthTxDirDefaultInXY = norm([txDirXDefault, txDirYDefault]);
txDirXDefaultNormed = txDirXDefault/lengthTxDirDefaultInXY;
txDirYDefaultNormed = txDirYDefault/lengthTxDirDefaultInXY;

%% Simulations
% Compute the beam alignment angles for each simulation.
numTxEleAngles = length(txEleAngels);
% The angle between the antenna and the direct link for the tx and the rx,
% respectively.
[beamAlignmentAnglesTx, beamAlignmentAnglesRx] ...
    = deal(nan(numPtsOnRoute, numTxEleAngles));
% Simulate the scenario with the TX elevation set according to txEleAngels.
for idxTxEleAngle = 1:numTxEleAngles
    txEleAngel = txEleAngels(idxTxEleAngle);
    % Note that the sph2cart function uses a different way for defining
    % azimuth angles.
    [txDirY, txDirX, txDirAlt] = sph2cart(usedAziAngleTxRad, ...
        deg2rad(txEleAngel), directionLength);
    lengthTxDirInXY = norm([txDirX, txDirY]);
    txDirXNormed = txDirX/lengthTxDirInXY;
    txDirYNormed = txDirY/lengthTxDirInXY;
    assert(approxEqual(txDirXNormed, txDirXDefaultNormed) ...
        && approxEqual(txDirYNormed, txDirYDefaultNormed), ...
        'Changing of elevatio should not change the resulting X and Y!')
    txDirXYAlt = [txDirX, txDirY, txDirAlt];
    % The RX will use the opposite direction as the TX.
    rxDirXYAlt = -txDirXYAlt;
    for idxPt = 1:numPtsOnRoute
        % The direct link from the RX to the TX (txLocVector -
        % rxLocVector).
        linkRx2Tx = [txX; txY; txAlt] ...
            - [rxXs(idxPt); rxYs(idxPt); rxAlts(idxPt)];
        % The direct link from the TX to the RX.
        linkTx2Rx = -linkRx2Tx;
        
        beamAlignmentAnglesTx(idxPt, idxTxEleAngle) ...
            = degBetweenTwo3DVectors(txDirXYAlt, linkTx2Rx);        
        beamAlignmentAnglesRx(idxPt, idxTxEleAngle) ...
            = degBetweenTwo3DVectors(rxDirXYAlt, linkRx2Tx);
    end
end

% A RX position on the route is good if both the TX and the RX are aligned
% well.
ratioOfGoodTxLocations ...
    = sum(beamAlignmentAnglesRx<=maxMisalignmentAngleAllowed ...
    & beamAlignmentAnglesTx<=maxMisalignmentAngleAllowed)' ...
    ./numPtsOnRoute;
[~, optiEleAngleTxIdx] = max(ratioOfGoodTxLocations);
optiEleAngleTx = txEleAngels(optiEleAngleTxIdx);
bestRatioOfGoodTxLocations = ratioOfGoodTxLocations(optiEleAngleTxIdx);

%% Plots
TX_MARKER = 'b^';
DIRCTION_MARKER = 'k-';
RX_ROUTE_MARKER = 'g.';
RX_ROUTE_START_PT_MARKER = 'r*';
GOOD_ANGLE_MARKER = 'k.';
NUM_TO_STR_FORMAT = '%.4f';

custNum2Str = @(s) num2str(s, NUM_TO_STR_FORMAT);

% Plot the results. We will generate 5 figures: one for the bird's-eye view
% of the scenario on a map; one for the simulated scenario in the 3D space;
% two for the (essentially 3D) plot of elevation angle vs. route location
% vs. beam alignment angle for the TX and the RX, respectively; and the
% last one for ratio of good RX locaitons vs. elevation angle.
hSimuFigs = cell(4,1);
% Scenario overview on a map.
hSimuFigs{1} = figure; hold on;
plot(txLatLonAlt(2), txLatLonAlt(1), TX_MARKER);
plot(rxLons, rxLats, RX_ROUTE_MARKER);
plot(rxLons(1), rxLats(1), RX_ROUTE_START_PT_MARKER);
legend('TX', 'RX route', 'RX start location');
plot_google_map('MapType', 'satellite');
xlabel('Lon'); ylabel('Lat');
xticklabels([]); yticklabels([]);
title('Scenario Overview on a Map');

% Scenario overview in 3D.
hSimuFigs{2} = figure; hold on;
plot3(txX, txY, txAlt, TX_MARKER);
% TX's default azimuth direction (pointing to the middle point of the route
% but with 0 elevation).
plot3([txX txX+txDirXDefault], [txY txY+txDirYDefault], ...
    [txAlt txAlt], DIRCTION_MARKER);
plot3(rxXs, rxYs, rxAlts, RX_ROUTE_MARKER);
plot3(rxXs(1), rxYs(1), rxAlts(1), RX_ROUTE_START_PT_MARKER);
legend('TX', 'TX''s azimuth direction', 'RX route', 'RX start location');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Alt (m)');
grid on; axis equal; view(3);
title('Scenario Overview in 3D');

% Simulation results: elevation angle vs. route location (distance from the
% RX start point) vs. beam alignment angle.
rxLocations = pdist2([rxXs(1), rxYs(1), rxAlts(1)], ...
    [rxXs, rxYs, rxAlts])';
% Generate the vector data for plotting.
beamAlignmentAnglesVectorTx = beamAlignmentAnglesTx(:);
beamAlignmentAnglesVectorRx = beamAlignmentAnglesRx(:);
rxLocationsVector = repmat(rxLocations(end:-1:1), numTxEleAngles, 1);
txEleAngelsVector = repmat(txEleAngels, numPtsOnRoute, 1);
txEleAngelsVector = txEleAngelsVector(:);
boolsGoodAnglesTx ...
    = beamAlignmentAnglesVectorTx<=maxMisalignmentAngleAllowed;
boolsGoodAnglesRx ...
    = beamAlignmentAnglesVectorRx<=maxMisalignmentAngleAllowed;

hSimuFigs{3} = figure; hold on;
plot3k([txEleAngelsVector rxLocationsVector beamAlignmentAnglesVectorTx]);
hGoodAngles = plot3(txEleAngelsVector(boolsGoodAnglesTx), ...
    rxLocationsVector(boolsGoodAnglesTx), ...
    90.*ones(sum(boolsGoodAnglesTx), 1), GOOD_ANGLE_MARKER);
view(0,90);
% legend(hGoodAngles, 'Good RX locations');
xlabel('TX elevation angle (degree)'); ylabel('RX location (m)');
zlabel('Beam alignment angle for TX');
grid on; axis tight;
title({'Simulation Results - Beam Alignment Angle for TX'; ...
    '(Good RX Positions are Blacked Out)';' '});

hSimuFigs{4} = figure; hold on;
plot3k([txEleAngelsVector rxLocationsVector beamAlignmentAnglesVectorRx]);
hGoodAngles = plot3(txEleAngelsVector(boolsGoodAnglesRx), ...
    rxLocationsVector(boolsGoodAnglesRx), ...
    90.*ones(sum(boolsGoodAnglesRx), 1), GOOD_ANGLE_MARKER);
view(0,90);
% legend(hGoodAngles, 'Good RX locations');
xlabel('TX elevation angle (degree)'); ylabel('RX location (m)');
zlabel('Beam alignment angle for RX');
grid on; axis tight;
title({'Simulation Results - Beam Alignment Angle for RX'; ...
    '(Good RX Positions are Blacked Out)';' '});

hSimuFigs{5} = figure; hold on;
hRatioOfGoodRxLocs = plot(txEleAngels, ratioOfGoodTxLocations, '.b--');
axis tight;
curYlim = ylim;
hOptimalEleAng = plot([optiEleAngleTx optiEleAngleTx], curYlim, 'g-.');
ylim(curYlim);
text(optiEleAngleTx, bestRatioOfGoodTxLocations,  ...
    ['(', custNum2Str(optiEleAngleTx), ', ', ...
    custNum2Str(bestRatioOfGoodTxLocations), ')'], ...
    'VerticalAlignment', 'top');
legend([hRatioOfGoodRxLocs, hOptimalEleAng], ...
    'Ratio of good RX locations', 'Optimal elevation angle');
xlabel('TX elevation angle (degree)');
ylabel('Ratio of good RX locations');
grid on;
title(['Optimal Elevation Angle: ', custNum2Str(optiEleAngleTx)]);

%% Save Plots
if exist('pathToSaveFig', 'var')
    figFileNamePrefix = 'air2GroundMeasSimulation';
    figNames = {'OverviewMap', 'Overview3D', 'BeamAlignAnglesTx', ...
        'BeamAlignAnglesRx', 'OptiEleAngle'};
    
    for idxFig = 1:length(figNames)
        saveas(hSimuFigs{idxFig}, [pathToSaveFig, figNames{idxFig}, '.fig']);
        saveas(hSimuFigs{idxFig}, [pathToSaveFig, figNames{idxFig}, '.png']);
    end
end

end
% EOF