// Set the geometry kernel
SetFactory("OpenCASCADE");

//---------------------------------------
// 2. Define Circle
//---------------------------------------
Circle(1) = {0, 0, 0, 200, 0, 2*Pi};
Curve Loop(3) = {1};


//---------------------------------------
// 2. Define Rectangles (capacitor plates)
//---------------------------------------
// Rectangle A
Rectangle(1) = {-50, 3, 0, 100, 4, 0};
Curve Loop(4) = {2, 3, 4, 5};

Rotate {{0, 0, 1}, {-50, 3, 0}, 0} { Surface{1}; }

// Rectangle B
Rectangle(2) = {-50, -3, 0, 100, 4, 0}; 
Curve Loop(5) = {6, 7, 8, 9};

Plane Surface(3) = {3, 4, 5};

//---------------------------------------
// 4. Subtract Rectangles from Circle
//---------------------------------------
BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; }

//---------------------------------------
// 5. Define Physical Groups
//---------------------------------------

//--- Physical Surfaces
Physical Surface("space", 10) = {3};                // Remaining domain after subtraction

//--- Physical Curves
Physical Curve("capacitorA", 11) = {2, 3, 4, 5};      // Rectangle A
Physical Curve("capacitorB", 12) = {6, 7, 8, 9};      // Rectangle B
Physical Curve("boundary", 20) = {1};                 // Circle perimeter

//--- Physical Points (all)
Physical Point("all_points", 30) = {
  1, 2, 3,      // Circle points
  4, 5, 6, 7,   // Rectangle A corners
  8, 9, 10, 11  // Rectangle B corners
};

// Define a distance field for mesh refinement
Field[1] = Distance;
Field[1].CurvesList = {2, 3, 4, 5, 6, 7, 8, 9};  // Rectangle curves
Field[1].NumPointsPerCurve = 200;

// Define threshold field to control mesh size based on distance to attractor
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.1; // Minimum element size near the attractor
Field[2].LcMax = 10;  // Maximum element size away from the attractor
Field[2].DistMin = 0.01;
Field[2].DistMax = 40;

Background Field = 2;


//---------------------------------------
// 6. Generate the Mesh
//---------------------------------------
Mesh 2;  // 2D mesh generation