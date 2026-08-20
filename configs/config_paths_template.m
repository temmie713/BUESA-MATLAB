function paths = config_paths_template()
%CONFIG_PATHS_TEMPLATE Template for local BUESA data/output paths.
%
% Copy this file to:
%   configs/config_paths_local.m
%
% Then rename the function inside the copied file to:
%   function paths = config_paths_local()
%
% IMPORTANT:
% - Do not commit config_paths_local.m.
% - Do not place participant-level data inside the public repository.

projectRoot = fileparts(fileparts(mfilename("fullpath")));

paths = struct();

% Source tables used for longitudinal label construction
paths.eegListFile    = fullfile("D:\PRIVATE_GARD", "brainwave_7175_oid_list.xlsx");
paths.diagnosisFile  = fullfile("D:\PRIVATE_GARD", "brainwave_dia_list.xlsx");

% Optional auxiliary cognitive-score export
paths.snsbFile       = fullfile("D:\PRIVATE_GARD", "snsb_total_fu.xlsx");

% Root directory containing authorized two-channel BEAM txt files
paths.beamRoot       = fullfile("D:\PRIVATE_GARD", "BEAM");

% Generated intermediate tables / features / analyses
paths.labelOutputDir   = fullfile(projectRoot, "outputs", "01_labels");
paths.featureOutputDir = fullfile(projectRoot, "outputs", "02_features");
paths.outputRoot       = fullfile(projectRoot, "outputs");

end
