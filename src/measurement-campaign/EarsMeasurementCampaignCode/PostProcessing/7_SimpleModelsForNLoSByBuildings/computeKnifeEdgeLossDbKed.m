function [ lossDb ] = computeKnifeEdgeLossDbKed( ...
    txLatLon, txHeightM, ...
    crossEdgeLatLons, crossLatLon, heightBuilding, ...
    rxLatLon, rxHeightM, lambdaInM, FLAG_DEBUG)
%COMPUTEKNIFEEDGELOSSDBKED Compute the knife edge loss in dB using the KED
%model.
%
% Input:
%    - txLatLon, txHeightM
%      The [lat, lon] and height (in m) for the TX.
%    - crossEdgeLatLons, crossLatLon, heightBuilding
%      The edge crossed with the TX and RX link in the form of [latStart,
%      lonStart; latEnd, lonEnd]; The [lat, lon] for the crossing point;
%      And the height of the building for the edge.
%    - rxLatLon, rxHeightM
%      The [lat, lon] and height (in m) for the RX.
%    - lambdaInM
%      Wavelength in m for the signal.
%    - FLAG_DEBUG
%      Optional. Set this to be true to generate debug plots.
%
% Output:
%    - lossDb
%      The resulting knife edge loss in dB. In this function, we will get
%      negative numbers to indicate loss.
%
% Yaguang Zhang, Purdue, 11/07/2017

if nargin<8
    % Set this to be true to generate debug plots.
    FLAG_DEBUG = false;
end

% Use the extended UTM system for the computation.
[txX, txY, txZone] = deg2utm(txLatLon(1), txLatLon(2));
[rxX, rxY, rxZone] = deg2utm(rxLatLon(1), rxLatLon(2));
[crossEdgeStartX, crossEdgeStartY, crossEdgeStartZone] ...
    = deg2utm(crossEdgeLatLons(1, 1), crossEdgeLatLons(1, 2));
[crossEdgeEndX, crossEdgeEndY, crossEdgeEndZone] ...
    = deg2utm(crossEdgeLatLons(2, 1), crossEdgeLatLons(2, 2));
[crossPtX, crossPtY, crossPtZone] ...
    = deg2utm(crossLatLon(1), crossLatLon(2));
assert(all([strcmp(txZone, rxZone), ...
    strcmp(rxZone, crossEdgeStartZone), ...
    strcmp(crossEdgeStartZone, crossEdgeEndZone), ...
    strcmp(crossEdgeEndZone, crossPtZone)]), ...
    'All GPS coordinates should be in the same UTM zone!');

% Find the intersection point.
linkLine = [txX, txY, txHeightM, rxX-txX, rxY-txY, rxHeightM-txHeightM];
edgeWallPoly = [crossEdgeStartX, crossEdgeStartY, 0; ...
    crossEdgeStartX, crossEdgeStartY, heightBuilding; ...
    crossEdgeEndX, crossEdgeEndY, heightBuilding; ...
    crossEdgeEndX, crossEdgeEndY, 0];
[interPt, boolInside] = intersectLinePolygon3d(linkLine, edgeWallPoly);

if(boolInside)
    d1 = norm([txX, txY, txHeightM] - [crossPtX, crossPtY, heightBuilding]);
    d2 = norm([crossPtX, crossPtY, heightBuilding] - [rxX, rxY, rxHeightM]);
    
    % The knife-edge loss for the top horizontal edge.
    %  hHor = norm([crossPtX, crossPtY, heightBuilding] - interPt);
    hHor = distancePointLine3d([crossPtX, crossPtY, heightBuilding], ...
        [interPt, interPt-[txX, txY, txHeightM]]);
    % The knife-edge loss for the closer horizontal edge.
    distMToEdges = nan(2,1);
    for idxEdge = 1:2
        distMToEdges(idxEdge) = ...
            lldistkm(txLatLon, crossEdgeLatLons(idxEdge,:)).*1000;
    end
    [~, idxCloserEdge] = min(distMToEdges);
    switch idxCloserEdge
        case 1
            crossPtVer = [crossEdgeStartX, crossEdgeStartY, interPt(3)];
        case 2
            crossPtVer = [crossEdgeEndX, crossEdgeEndY, interPt(3)];
        otherwise
            error('There should be one closer verticle edge!')
    end
    % hVer = norm(crossPtVer - interPt);
    hVer = distancePointLine3d(crossPtVer, ...
        [interPt, interPt-[txX, txY, txHeightM]]);
    
    % Use the smaller h to get a smaller loss. Intuitively, this means we
    % only consider the dominant knife edge path (hor / ver).
    h = min(hHor, hVer);
    
    %  Ref:
    %    Wireless Communications - Principles and Practice by T. Rappaport
    % nu = h.*sqrt(2.*(d1+d2)./lambdaInM./d1./d2);
    %  intFct = @(t) exp(-1i.*(pi./2).*(t.^2));
    % F_nu = ((1+1i)./2.*integral(intFct, nu, inf));
    %  lossDb = 20.*log10(abs(F_nu));
    
    %  Ref:
    %    Electromagnetic Waves and Antennas by S. Orfanidis
    
    
    % Notes:
    %        1/F = 1/r1 + 1/r2 or F = r1*r2/(r1+r2)
    % F = d1.*d2./(d1+d2);
    %        v = sqrt(2/lambda*F) * b, where b = clearance distance from
    %        edge.
    % v = sqrt(2./lambdaInM.*F) .* h;
    %        This calculates D = (F(v) + (1-j)/2)/(1-j), where F(v) = C(v)
    %        - jS(v) = complex Fresnel function and F(v) is calculated
    %        using fcs(v).
    % D = diffr(v);
    %        diffraction loss is L = -20*log10(abs(D))
    % lossDb = 20.*log10(abs(D));
    
    % We will use both refs. The diffr(v) is F(-nu).
    nu = h.*sqrt(2.*(d1+d2)./lambdaInM./d1./d2);
    F_nu = diffr(-nu);
    lossDb = 20.*log10(abs(F_nu));
else
    lossDb = 0;
end

if FLAG_DEBUG
    hDebugFif = figure; hold on;
    numGpsSamps = evalin('base', 'numGpsSamps');
    pathLossesWithGpsHol = evalin('base', 'pathLossesWithGpsHol');
    for idx = 1:numGpsSamps
        curPathLossWithGps = pathLossesWithGpsHol(idx,:);
        % Conti track.
        [x,y,zone] = deg2utm(curPathLossWithGps(2), curPathLossWithGps(3));
        if strcmp(zone, txZone)
            hTrack = plot3(x, y, rxHeightM, '.-', 'Color', ones(1,3)*0.5);
        end
    end
    
    hWall = patch([crossEdgeStartX, crossEdgeStartX, crossEdgeEndX, crossEdgeEndX], ...
        [crossEdgeStartY, crossEdgeStartY, crossEdgeEndY, crossEdgeEndY], ...
        [0, heightBuilding, heightBuilding, 0], 'red', 'FaceAlpha', 0.9);
    hTx = plot3(txX, txY, txHeightM, '^k', 'MarkerSize', 5, ...
        'LineWidth', 0.5, 'MarkerFaceColor', 'b');
    hRx = plot3(rxX, rxY, rxHeightM, 'vk', 'MarkerSize', 5, ...
        'LineWidth', 0.5, 'MarkerFaceColor', 'g');
    lineWidthPts = 1.5;
    hLoS = plot3([txX,rxX], [txY,rxY], [txHeightM,rxHeightM], 'b-.', ...
        'LineWidth', 1);
    if hHor < hVer
        % The interPt is closer to the top horizontal edge.
        crossPt = [crossPtX, crossPtY, heightBuilding];
        hCross = plot3(crossPtX, crossPtY, heightBuilding, ...
            'ok', 'MarkerSize', 4, ...
        'LineWidth', 1, 'MarkerFaceColor', 'g');
    else
        crossPt = crossPtVer;
        hCross = plot3(crossPtVer(1), crossPtVer(2), crossPtVer(3), ...
            'ok', 'MarkerSize', 4, ...
        'LineWidth', 1, 'MarkerFaceColor', 'g');
    end
    hShortestPath = plot3([txX,crossPt(1)], [txY,crossPt(2)], ...
        [txHeightM,crossPt(3)], 'k-', ...
        'LineWidth', 1);
    plot3([rxX,crossPt(1)], [rxY,crossPt(2)], ...
        [rxHeightM,crossPt(3)], 'k-', ...
        'LineWidth', 1);
    % hInters = plot3(interPt(1), interPt(2), interPt(3), 'ok', 'MarkerSize', 4, ...
    %    'LineWidth', lineWidthPts, 'MarkerFaceColor', 'w');
    axis equal;
    legend([hTx, hTrack, hRx, hWall, hShortestPath, hCross, hLoS], 'TX', ...
        'GPS Track', 'RX Location of Interest', ...
        'Building Wall', 'Shortest Path', ...
        'Crossing Point',  ...
        'Imaginary LoS Path', 'Location', 'northwest');
    view(3); xlabel('x (m)'); ylabel('y (m)'); zlabel('height (m)'); grid on;
    disp(['lossDb = ', num2str(lossDb)]); 
    set(gca, 'FontWeight', 'bold');
    disp('Debug figure generated. Press any key to continue...')
    pause;
    close(hDebugFif);
end

end
%EOF