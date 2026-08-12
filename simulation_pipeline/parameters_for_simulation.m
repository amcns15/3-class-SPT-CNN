trj.trajectoryFile = '-';
trj.reactionFile = '-';
trj.degradedName   = '-1';
trj.timeScale      = 1;
trj.voxelSize      = 1;
trj.speciesNames   = {'-'};
trj.D              = [0];

output.resultFile  = '-';
camera.xrange_px   = 16;
camera.yrange_px   = 44;
camera.pixLength   = 0.05;
camera.A(1,:)=[0           -1           0];
camera.A(2,:)=[1           0           0];
camera.A(3,:)=[0           0           1];
camera.b=[ 0.4712      1.1277    -0.0102]'; %(note transpose!)
sample.dt          = 0.009;
sample.tE          = 0.001;

camera.alpha = 0.005;
camera.offset = 100;
camera.sigmaReadout = 6;

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

sample.dt = 0.01;
sample.tE = 0.005;

% activation.t1 = 0;
% activation.ta = 0.1;
% activation.Pa = 0.3;
% activation.td = 0;
% activation.ka = [0 0];
activation.type = 'SM_photoActivation_instant';

baseIntensity.intensity = 2e5;
baseIntensity.type = 'SM_activationIntensity_uniform';

photophys.bleach_time = 5000;
photophys.type = 'SM_fluo_only_bleach';

psf.type = 'SM_psf_constant_gaussian';
psf.sigma = 0.1;

background.photons_per_pixel=1;
background.type='SM_bg_constant';