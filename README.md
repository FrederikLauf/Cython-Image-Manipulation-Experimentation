# Cython Image Manipulation Experimentation

Feasibility study for the use of Cython for color editing RGB images.

## Current features:
### Desaturate color towards arbitrary tone
Turn all color vectors towards a chosen color vector by a chosen fraction of the original angle.
### Saturate color
Turn all color vectors away from the grey direction (angle = - \<factor\> * (\<constant\> - \<original angle\>)).
### Color brightness
Scale color components independently.
### Color calibration
Turn all color vectors araound a chosen axis by a chosen angle.
