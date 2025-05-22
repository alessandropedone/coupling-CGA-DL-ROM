// Set the geometry kernel
SetFactory("OpenCASCADE");

//---------------------------------------
// Rectangles
//---------------------------------------

// Rectangle 1 (top)
Point(1) = {-49.67777777777778, 1.461111111111111, 0, 1.0};
Point(2) = {49.67777777777778, 1.461111111111111, 0, 1.0};
Point(3) = {49.67777777777778, 4.816666666666667, 0, 1.0};
Point(4) = {-49.67777777777778, 4.816666666666667, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Rotate { {0, 0, 1}, {0,  1.1388888888888888,  0}, -0.002 } {
  Surface{1};
}

// Rectangle 2 (bottom)
Point(5) = {-50,-1.1388888888888888, 0, 1.0};
Point(6) = { 50,-1.1388888888888888, 0, 1.0};
Point(7) = { 50,-5.138888888888889, 0, 1.0};
Point(8) = {-50,-5.138888888888889, 0, 1.0};

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

// Subtract Rectangles from Circle
BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; }

//---------------------------------------
// Transfinite Lines and Surface
//---------------------------------------
r = 7;
Transfinite Line {1, 3} = 50*r Using Progression 1;
Transfinite Line {2, 4} = 2*r Using Progression 1;
Transfinite Line {5, 7} = 50*r Using Progression 1;
Transfinite Line {6, 8} = 2*r Using Progression 1;
Transfinite Line {9, 10, 11, 12} = 20 Using Progression 1;

//---------------------------------------
// Define Physical Groups
//---------------------------------------                

//--- Physical Curves
// Physical groups for boundaries
Physical Line("force_segment", 10) = {1};
Physical Line("upper_plate", 11) = {2, 3, 4};
Physical Line("lower_plate", 12) = {5, 6, 7, 8};
Physical Line("boundary", 20) = {9, 10, 11, 12};

//--- Physical Surfaces
Physical Surface("space", 30) = {3};

//---------------------------------------
// 6. Generate the Mesh
//---------------------------------------
Mesh 2;  // 2D mesh generation