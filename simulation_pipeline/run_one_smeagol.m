function run_one_smeagol( ...
    smeagolSetup, ...
    templateRuninput, ...
    trajectoryFile, ...
    reactionFile, ...
    outputMatFile, ...
    species)
    %diffusion_coef ...
    
    
    % Add SMeagol to the MATLAB path
    run(smeagolSetup);

    opt = SM_getOptions(templateRuninput); % load existing settings

    %Replace input and output files
    opt.trj.trajectoryFile = trajectoryFile;
    opt.trj.reactionFile = reactionFile;
    opt.trj.speciesNames = species;
    %opt.trj.D = diffusion_coef;
    opt.trj.D = 0
    opt.output.resultFile = outputMatFile;

    % is this needed?
    opt.runinputroot = '';

    % TIFF output
    opt.output.writeTifMovie = 1;
    opt.output.plotTifMovie = 1;

    % Run simulation
    SM_runsimulation(opt);
end


