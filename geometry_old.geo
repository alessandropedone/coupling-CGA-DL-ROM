//+
SetFactory("OpenCASCADE");
Circle(1) = {1, 1, 0, 1, 0, 2*Pi};
//+
Rectangle(1) = {0.5, 0.8, 0, 1, 0.02, 0};
//+
Rectangle(2) = {0.6, 1.3, 0, 0.8, 0.02, 0};
//+
Curve Loop(3) = {1};
//+
Curve Loop(4) = {8, 9, 6, 7};
//+
Curve Loop(5) = {4, 5, 2, 3};
//+
Plane Surface(3) = {3, 4, 5};
//+
Physical Curve("boundary", 10) = {1};
//+
Physical Curve("capacitorA", 11) = {8, 9, 6, 7};
//+
Physical Curve("capacitorB", 12) = {4, 5, 2, 3};
//+
BooleanDifference{ Surface{3}; Delete; }{ Surface{2}; Surface{1}; Delete; }
//+
Physical Surface("space", 13) = {3};
//+
Physical Surface(" space", 13) -= {3};
//+
Physical Surface(14) = {3};
//+
Physical Surface("space", 15) = {3};
//+
Physical Surface("space", 15) += {3};
//+
Physical Surface(" space", 15) -= {3};
//+
Physical Surface(14) -= {3};
//+
Physical Surface(" space", 15) -= {3};
//+
Physical Surface(16) -= {3};
//+
Physical Surface("space", 35) = {3};
//+
Physical Point("capacitor", 36) = {9, 8, 7, 6, 5, 2, 3, 4};
//+
Show "*";
//+
Physical Point("boundarypoint", 37) = {1};
//+
Physical Surface(" space", 35) -= {3};
//+
Physical Surface("space", 38) = {3};
