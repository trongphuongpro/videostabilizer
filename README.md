### Stabilizer for living video stream
#### Requirements:
- OpenCV 3.x
- NumPy

#### Usages:
- initialization: stabilizer = **VideoStabilizer**(videosource, *, size=(640,480), processVar=0.1, measVar=2);
- get stabilized frame: successStatus, originalFrame, stabilizedFrame = stabilizer.**read**().

**Sample program**: test.py
