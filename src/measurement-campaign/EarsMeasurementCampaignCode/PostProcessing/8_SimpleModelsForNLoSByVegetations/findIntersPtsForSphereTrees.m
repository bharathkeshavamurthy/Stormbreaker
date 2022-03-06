function [intersPtsXYZs, intersPtsLatLons, widthsInM, totalWidthInM] ...
    = findIntersPtsForSphereTrees(txLatLon, txHeightM, ...
    rxLatLon, rxHeightM, trees)
%FINDINTERSPTSFORSPHERETREES Find the intersect points for the TX & RX link
%with trees modeled by spheres.
%
% Input:
%    - txLatLon, txHeightM
%      The [lat, lon] and height (in m) for the TX.
%    - rxLatLon, rxHeightM
%      The [lat, lon] and height (in m) for the RX.
%    - trees
%      The parameter matrix for the trees: [lat, lon, folaigeRadiusInMeter,
%      foliageCenterHeightInMeter]
%
% Output:
%    - intersPtsXYZs
%      The intersection points. Each row cooresponds to one tree in the
%      form of [x1 y1 z1 ; x2 y2 z2].
%    - intersPtsLatLons
%      The lat and lon for the intersection points. Each row cooresponds to
%      one tree in the form of [lat1 lon1 ; lat2 lon2].
%    - widthsInM
%      A colom vector for the widths of foliage for the trees in meter.
%    - totalWidthInM
%      The total foliage width in meter.
%
% Yaguang Zhang, Purdue, 11/07/2017

[numTrees, ~] = size(trees);
intersPtsXYZs = nan(numTrees, 6);
intersPtsLatLons = nan(numTrees, 4);
widthsInM = nan(numTrees, 1);

[txX, txY, txZone] = deg2utm(txLatLon(1), txLatLon(2));
[rxX, rxY, rxZone] = deg2utm(rxLatLon(1), rxLatLon(2));
assert(strcmp(txZone, rxZone), ...
    'The UTM zones for the TX and the RX should be the same!');

line = [txX, txY, txHeightM, rxX-txX, rxY-txY, rxHeightM-txHeightM];
for idxTree = 1:numTrees
    [treeX, treeY, treeZone] = deg2utm(trees(idxTree, 1), ...
        trees(idxTree, 2));
    assert(strcmp(txZone, treeZone), ...
    'The UTM zones for the TX and the tree should be the same!');

    % Find and record the intersection points.
    sphere = [treeX, treeY, trees(idxTree, 4), trees(idxTree, 3)];
    points = intersectLineSphere(line, sphere)';    
    intersPtsXYZs(idxTree, :) = points(:)';
    [intersPtsLatLons(idxTree, 1), intersPtsLatLons(idxTree, 2)] ...
        = utm2deg(intersPtsXYZs(idxTree, 1),intersPtsXYZs(idxTree, 2),txZone);
    [intersPtsLatLons(idxTree, 3), intersPtsLatLons(idxTree, 4)] ...
        = utm2deg(intersPtsXYZs(idxTree, 4),intersPtsXYZs(idxTree, 5),txZone);
    widthsInM(idxTree) = norm(points(:,1)-points(:,2));
end

totalWidthInM = sum(widthsInM(~isnan(widthsInM)));

end
% EOF