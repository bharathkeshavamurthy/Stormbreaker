/**
  Copyright (C) 2012-2020 by Autodesk, Inc.
  All rights reserved.

  Flow waterjet post processor configuration.

  $Revision: 42945 3b392c2b5da742831c838c194f3f14fb822b0add $
  $Date: 2020-09-22 12:12:10 $
  
  FORKID {F61954EF-5A29-4B93-93E7-870BC2786880}
*/

description = "Flow Waterjet ORD";
vendor = "Flow";
vendorUrl = "http://www.flowwaterjet.com";
legal = "Copyright (C) 2012-2020 by Autodesk, Inc.";
certificationLevel = 2;
minimumRevision = 39000;

longDescription = "Post for Flow Waterjets using software Version 5 or 6. V5=2-axis, V6=3-axis. " +
"V5: Manually set nozzle height. V6: Nozzle height in NC program set by Top Height attribute. " + EOL +
"Feed percentage set by Cutting Mode quality in tool dialog (auto=60, high=20, medium=40, low=100)";

extension = "ord";
setCodePage("ascii");

capabilities = CAPABILITY_JET;
tolerance = spatial(0.002, MM);

minimumChordLength = spatial(0.25, MM);
minimumCircularRadius = spatial(0.01, MM);
maximumCircularRadius = spatial(1000, MM);
minimumCircularSweep = toRad(0.01);
maximumCircularSweep = toRad(90);
allowHelicalMoves = false;
allowedCircularPlanes = 1 << PLANE_XY; // allow only XY circular motion

// formatVersion:
// version 6 should be user selectable, but now use version 5
// version 5 does not have z moves, so it will be safer for inexperienced users
// version 5 does not have the mysterious parameter after the cutter comp value

properties = {
  useHSMFeedrates: false,
  maximumFeedrateIPM: 700, // specifies the rapid traverse maximum
  formatVersion: "6"  // format version 5 or 6
};

// user-defined property definitions
propertyDefinitions = {
  useHSMFeedrates: {title:"Use HSM feedrates", description:"Specifies whether to output the feedrates from HSM.", type:"boolean"},
  maximumFeedrateIPM: {title: "Maximum feedrate (IPM)", description: "Sets the maximum feedrate in IPM.", type: "integer"},
  formatVersion: {title: "Flow control software version",
    description: "V5 outputs XY, V6 outputs XYZ",
    group: 1,
    type:"enum",
    values:[
      {title:"V5", id:"5"},
      {title:"V6", id:"6"}
    ]
  }
  
};

// use fixed width instead
var xyzFormat = createFormat({decimals:4, trim:false});
var integerFormat = createFormat({decimals:0});

// fixed settings
var thickness = 0;
var arcCounter = 0;
var lineCounter = -1;

// collected state
var useVersion6 = false;
var arcFinish = undefined;

// override radius compensation
var compensationOffset = 0; // center compensation

var etchOperation = false; // though-cut unless set to true
var forceOutput = false;

/**
  Writes the specified block.
*/
function writeBlock() {
  writeWords(arguments);
}

var FIELD = "                    ";

/** Make sure fields are aligned. */
function f(text) {
  var length = text.length;
  if (length > 10) {
    return text;
  }
  return FIELD.substr(0, 10 - length) + text;
}

/** Make sure fields are aligned. */
function fi(text, size) {
  var length = text.length;
  if (length > size) {
    return text;
  }
  return FIELD.substr(0, size - length) + text;
}

function onOpen() {
  useVersion6 = properties.formatVersion == "6";
  unit = IN; // only inch mode is supported

  redirectToBuffer(); // buffer the entire program to be able to count the linear and circular moves
  setWordSeparator("");

  { // stock - workpiece
    var workpiece = getWorkpiece();
    var delta = Vector.diff(workpiece.upper, workpiece.lower);
    if (delta.isNonZero()) {
      // thickness = (workpiece.upper.z - workpiece.lower.z);
    }
  }

  if (getNumberOfSections() > 0) {
    var firstSection = getSection(0);

    var remaining = firstSection.workPlane;
    if (!isSameDirection(remaining.forward, new Vector(0, 0, 1))) {
      error(localize("Tool orientation is not supported."));
      return;
    }
    setRotation(remaining);

    if (!useVersion6) {
      var originZ = firstSection.getGlobalZRange().getMinimum(); // the cutting depth of the first section

      for (var i = 0; i < getNumberOfSections(); ++i) {
        var section = getSection(i);
        var z = section.getGlobalZRange().getMinimum(); // contour Z of the each section
        if (xyzFormat.getResultingValue(z) != xyzFormat.getResultingValue(originZ)) {
          error(localize("You are trying to machine at multiple depths which is not allowed."));
          return;
        }
      }
    }
  }
  forceOutput = true;// force out first line
}

function onSection() {
  var remaining = currentSection.workPlane;
  if (!isSameDirection(remaining.forward, new Vector(0, 0, 1))) {
    error(localize("Tool orientation is not supported."));
    return;
  }
  setRotation(remaining);

  etchOperation = false;
  if (currentSection.getType() == TYPE_JET) {
    switch (tool.type) {
    case TOOL_WATER_JET:
      break;
    default:
      error(localize("The CNC does not support the required tool."));
      return;
    }
    switch (currentSection.jetMode) {
    case JET_MODE_THROUGH:
      break;
    case JET_MODE_ETCHING:
      etchOperation = true;
      break;
    case JET_MODE_VAPORIZE:
      error(localize("Vaporize is not supported by the CNC."));
      return;
    default:
      error(localize("Unsupported cutting mode."));
      return;
    }
  } else if (currentSection.getType() == TYPE_MILLING) {
    warning(localize("Milling toolpath will be used as waterjet through-cutting toolpath."));
  } else {
    error(localize("CNC doesn't support the toolpath."));
    return;
  }

  var initialPosition = getFramePosition(currentSection.getInitialPosition());
  onExpandedRapid(initialPosition.x, initialPosition.y, initialPosition.z);
}

function onParameter(name, value) {
}

function writeLinear(x, y, z, feed, column7) {
  var flag = false;
  if (useVersion6) {
    flag = xyzFormat.areDifferent(x, getCurrentPosition().x) ||
           xyzFormat.areDifferent(y, getCurrentPosition().y) ||
           xyzFormat.areDifferent(z, getCurrentPosition().z);
  } else {
    flag = xyzFormat.areDifferent(x, getCurrentPosition().x) ||
           xyzFormat.areDifferent(y, getCurrentPosition().y);
  }

  if (flag || forceOutput) {
    if (useVersion6) {
      writeBlock(
        f(xyzFormat.format(x)), ",",
        f(xyzFormat.format(y)), ",",
        f(xyzFormat.format(z)), ",",
        fi(integerFormat.format(0), 2), ",", // linear
        fi(integerFormat.format(feed), 5), ",",
        fi(integerFormat.format(compensationOffset), 2), ",", // left, center, right
        fi(integerFormat.format(column7), 2) // TAG: seen -2..2 - unknown
      );
    } else {
      writeBlock(
        f(xyzFormat.format(x)), ",",
        f(xyzFormat.format(y)), ",",
        fi(integerFormat.format(0), 2), ",",
        fi(integerFormat.format(feed), 5), ",",
        fi(integerFormat.format(compensationOffset), 2)
      );
    }
    ++lineCounter;
    forceOutput = false;
  }

}

function finishArcs() {
  if (arcFinish) {
    // complete circular motion with output of destination values
    forceOutput = true;
    writeLinear(arcFinish.x, arcFinish.y, arcFinish.z, arcFinish.feed, 2);
    arcFinish = undefined;
  }
}

function onRapid(x, y, z) {
  finishArcs();
  writeLinear(x, y, z, 0, 0); // non-cutting
}

function onLinear(x, y, z, feed) {
  finishArcs();
  // skip output if next move is an arc. OnCircular record has this move destination XY
  if (getNextRecord().getType() != RECORD_CIRCULAR) {
    writeLinear(x, y, z, power ? getFeedInPercent(feed) : 0, 2);
  }
}

function onPower(power) {
}

function onCircular(clockwise, cx, cy, cz, x, y, z, feed) {
  // spirals are not allowed - arcs must be < 360deg
  // fail if radius compensation is changed for circular move

  // syntax is; start X, start y, arc direction, feed, comp, cx, cy
  // end X, end Y, if following move is an arc, is the start x, start y, of the line
  // end X, end Y, if following move is a line, is output in OnLinear
  // using the arcFinish flag
  
  circularICode = (clockwise ? 1 : -1);

  if ((getCircularPlane() != PLANE_XY) || isHelical()) {
    linearize(tolerance);
  }
  var p = getCurrentPosition();
  if (useVersion6) {
    writeBlock(
      f(xyzFormat.format(p.x)), ",",
      f(xyzFormat.format(p.y)), ",",
      f(xyzFormat.format(p.z)), ",",
      fi(integerFormat.format(circularICode), 2), ",", // arc cw/ccw
      fi(integerFormat.format(power ? getFeedInPercent(feed) : 0), 5), ",",
      fi(integerFormat.format(compensationOffset), 2), ",", // left, center, right
      fi(integerFormat.format(2), 2), ",", // TAG: seen -2..2 - unknown
      f(xyzFormat.format(cx)), ",",
      f(xyzFormat.format(cy)), ",",
      f(xyzFormat.format(0)) // PLANE_XY only
    );
  } else {
    writeBlock(
      f(xyzFormat.format(p.x)), ",",
      f(xyzFormat.format(p.y)), ",",
      fi(integerFormat.format(circularICode), 2), ",", // arc cw/ccw
      fi(integerFormat.format(power ? getFeedInPercent(feed) : 0), 5), ",",
      fi(integerFormat.format(compensationOffset), 2), ",", // left, center, right
      f(xyzFormat.format(cx)), ",",
      f(xyzFormat.format(cy))
    );
  }
  ++arcCounter;

  // save destination values to complete arc move
  arcFinish = {x:x, y:y, z:z, feed:(power ? getFeedInPercent(feed) : 0)};

}

function getFeedInPercent(feed) {
  var feedPercent;
  if ((properties.maximumFeedrateIPM > 0) && properties.useHSMFeedrates) {
    // use HSM feedrates
    // 1 - 99 %
    feedPercent = Math.min(Math.ceil(Math.min(properties.maximumFeedrateIPM, feed) / properties.maximumFeedrateIPM * 100), 99);
  } else {
    // use fixed feedrates per quality selection
    switch (currentSection.quality) {
    case 1:
      // high quality
      feedPercent = 20; // very slow, cut surface excellent
      break;
    case 2:
	 // medium quality
      feedPercent = 60; // moderate, cut surface moderate
      break;
    case 3:
      feedPercent = 100; // fast, cut surface slightly rough
      break;
    default:
      feedPercent = 40; // slow, cut surface good
    }
  }
  return feedPercent;
}

function onRadiusCompensation() {
  switch (radiusCompensation) {
  case RADIUS_COMPENSATION_LEFT:
    compensationOffset = -1;
    break;
  case RADIUS_COMPENSATION_RIGHT:
    compensationOffset = 1;
    break;
  default:
    compensationOffset = 0; // center compensation
  }
}

function onCycle() {
  error(localize("Canned cycles are not supported."));
}

function onSectionEnd() {
  finishArcs();
  compensationOffset = 0; // center compensation
}

function onClose() {
  if (isRedirecting()) {
    var mainProgram = getRedirectionBuffer(); // TAG: no need for redirection
    closeRedirection();
    writeln("// This file was created by FlowMaster(R), which is proprietary to Flow International Corporation. " + lineCounter + " " + arcCounter);
    if (useVersion6) {
      writeln("VER 6.00");
    }
    writeln("// Created by Autodesk HSM");
    if (programComment) {
      writeln("// " + programComment);
    }
    write(mainProgram);
  }
}
