## GPAS: Gaussian Process Adaptive Sampling

*This is currently under construction.  
Please contact Lauren Miller (laurenm[@]berkeley[dot]edu) for questions.*

This is demonstration code for [Tumor  Localization  using  Automated  Palpation  withGaussian  Process  Adaptive  Sampling](http://www.lmiller.me/uploads/5/2/3/4/52340331/2016_case_palpationstrategy__7_.pdf). The paper will be presented at [IEEE CASE 2016](http://case2016.org).

#### Dependencies

In addition to numpy, scipy, matplotlib, this code depends on:
- gPy
- shapely

#### Running Demo Files
To run a single simulated experiment, run runPhase2.py

To run a batch of simulated experiments, varying noise and bias levels, run scaffoldPhase2.py 

To run an experiment on the robot, edit configs in runPhase2.py and run (needs ROS)

