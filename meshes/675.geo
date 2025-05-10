// Set the geometry kernel
SetFactory("OpenCASCADE");

//---------------------------------------
// Rectangles
//---------------------------------------

// Rectangle 1 (top)
Point(1) = {-49.56666666666667, 1.1388888888888888, 0, 1.0};
Point(2) = {49.56666666666667, 1.1388888888888888, 0, 1.0};
Point(3) = {49.56666666666667, 4.272222222222222, 0, 1.0};
Point(4) = {-49.56666666666667, 4.272222222222222, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Rotate { {0, 0, 1}, {0,  1.1388888888888888,  0}, 0.002 } {
  Surface{1};
}

// Rectangle 2 (bottom)
Point(5) = {-49.56666666666667, -1.1388888888888888, 0, 1.0};
Point(6) = {49.56666666666667, -1.1388888888888888, 0, 1.0};
Point(7) = {49.56666666666667, -4.272222222222222, 0, 1.0};
Point(8) = {-49.56666666666667, -4.272222222222222, 0, 1.0};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2}; 


//---------------------------------------
// Circle
//---------------------------------------

// Define points
Point(9) = {0, 0, 0};        // Center
Point(10) = {200, 0, 0};     // Start point
Point(11) = {0, 200, 0};     // 90 degrees
Point(12) = {-200, 0, 0};    // 180 degrees
Point(13) = {0, -200, 0};    // 270 degrees

// Define circle arcs (each needs start, center, end)
Circle(9) = {10, 9, 11};
Circle(10) = {11, 9, 12};
Circle(11) = {12, 9, 13};
Circle(12) = {13, 9, 10};

// Define curve loop and surface if needed
Curve Loop(3) = {9, 10, 11, 12};
Surface(3) = {3};

//---------------------------------------
// Define Physical Groups
//---------------------------------------                

//--- Physical Curves
// Physical groups for boundaries
Physical Line("force_segment", 10) = {1};
Physical Line("upper_plate", 11) = {2, 3, 4};
Physical Line("lower_plate", 12) = {5, 6, 7, 8};
Physical Line("boundary", 20) = {9, 10, 11, 12};

// Subtract Rectangles from Circle
BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; }
//--- Physical Surfaces
Physical Surface("space", 30) = {3};


//---------------------------------------
// Fine mesh near the plates
//--------------------------------------- 

// Define a distance field for mesh refinement
Field[1] = Distance;
Field[1].CurvesList = {1, 2, 3, 4, 5, 6, 7, 8};  // Rectangle curves
Field[1].NumPointsPerCurve = 200;

c = 0.3;
// Define threshold field to control mesh size based on distance to attractor
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = c; // Minimum element size near the attractor
Field[2].LcMax = 100*c;  // Maximum element size away from the attractor
Field[2].DistMin = 0.01;
Field[2].DistMax = 40;

Background Field = 2;

//---------------------------------------
// 6. Generate the Mesh
//---------------------------------------
Mesh 2;  // 2D mesh generation