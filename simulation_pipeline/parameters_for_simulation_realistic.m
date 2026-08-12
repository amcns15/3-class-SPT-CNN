% trj: information about the input data.
trj.reactionFile='-';
trj.trajectoryFile='-';
trj.degradedName='-1';
trj.timeScale=1;
trj.voxelSize=1;
trj.speciesNames{1}='-';
trj.D=[0];

% output: what to output, and where.
output.resultFile='-';
output.writeTifMovie   = true;
output.plotTifMovie    = false;
output.showPhotons      = false;
output.showEmitters     = false;
output.plotTrj          = false;
output.movieLength      = 100;
output.maxFrames        = 100;
output.plotTrjZRange    = [-5 5];
output.movieFormat      = 'tiff';
output.movieOptsImwrite  = {};

% sample: parameters describing illumination and image capture. SMeagol
% basically assumes that illumination and aquisition coincide, but
% continuous illumination can be modeled by an appropriate choice of
% photophysics-parameters.
sample.dt=0.01;
sample.tE=0.01;

% activation: parameters describing the fluorophore activation process.
activation.type='SM_photoActivation_instant';

% baseIntensity: every fluorescent group in the simulation has a basic
% emission intensity (photons/time) during illumination, which can vary
% from molecule to molecule, as determined at activation by these
% parameters.
baseIntensity.intensity=20000;
baseIntensity.type='SM_activationIntensity_uniform';

% photophys: parameters describing the dynamics of blinking and bleaching
% in terms of a Markov process (independent of the diffusive states
% described by the input trajectories).
photophys.emissionFactor=[ 1           0];
photophys.kb{1}=0;
photophys.kb{2}=0;
photophys.Q{1}(1,:)=[0          50];
photophys.Q{1}(2,:)=[200           0];
photophys.Q{2}(1,:)=[0          50];
photophys.Q{2}(2,:)=[200           0];
photophys.type='SM_fluo_full_markov';

% psf: parameters to simulate the microscope point-spread-function, i.e.,
% the (stochastic) map from the position of a fluorophore as it emits a
% photon to the position on the camera chip at which that photon is
% detected.
psf.sigma=0.05;
psf.type='SM_psf_constant_gaussian';

% camera: these parameters describe a) The region of interest (ROI), i.e.,
% the size, shape, and location of the region imaged by the camera, and b)
% the noise properties of the EMCCD chip.
% (a) is described in terms of the size and number of active camera pixels,
% plus a linear transformation of simulated coordinates x to
% camera-centered coordinates y, given by y = (voxelsize)*A*x+b, where A is
% a 3*3 matrix, and b is a 3*1 vector. 
% (b) is parameterized in terms of the offset, readout noise (standard
% deviation), and EM gain (average number of photons per camera count). We
% use the model described in the Mortensen et al. (Nat Meth 7, 377–381,
% 2010, doi: 10.1038/nmeth.1447).
camera.alpha=0.00025;
camera.offset=100;
camera.sigmaReadout=6;
camera.pixLength=0.05;
camera.xrange_px=16;
camera.yrange_px=44;
camera.A(1,:)=[0          -1           0];
camera.A(2,:)=[1           0           0];
camera.A(3,:)=[0           0           1];
camera.b=[ 0.4712      1.1277     -0.0102]'; %(note transpose!)

% background: parameters to describe how the noisy image background is to
% be generated.
background.photons_per_pixel=1;
background.type='SM_bg_constant';


