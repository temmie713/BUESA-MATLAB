%% 05_validate_fixed_cabess_signature.m
% Repeated subject-level Bayesian validation of the fixed CaBESS signature.
% Bayesian-only benchmark for BUESA EEG progression experiments
% Label compatibility:
%   The progression-subtype priority reproduces the final manuscript cohort so that the
%   primary manuscript task contains 325 non-decline and 45 decline records.
%
% Purpose
%   Evaluate the fixed seven-feature CaBESS signature and the all-feature
%   Bayesian baseline using the same repeated subject-level splits. The
%   Bayesian classifier uses a Gaussian prior, MAP estimation, and a Laplace
%   posterior approximation.
%
% Leakage control
%   - Cross-validation folds are generated at the participant level using ObjectID.
%   - All records from the same participant remain in the same fold.
%   - Missing-value imputation and preprocessing parameters are estimated from
%     the training fold only.
%   - Top-K feature ranking is computed from the training fold only.
%   - Class weights are computed from the training fold only.
%   - Test-fold records are used only for final evaluation.
%
% Main outputs
%   - BayesianOnly_TaskClassCounts.xlsx
%   - BayesianOnly_RawResults.xlsx
%   - BayesianOnly_Summary.xlsx
%   - BayesianOnly_BestByBalancedAccuracy.xlsx
%   - BayesianOnly_FeatureSelectionStability.xlsx
%   - BayesianOnly_SelectedFeatures_ByConfig.xlsx

clc; clear; close all;

%% =========================================================
% 0. Project path and configuration
%% =========================================================

projectRoot = fileparts(mfilename("fullpath"));

addpath(fullfile(projectRoot, "configs"));

if exist("config_paths_local", "file")
    paths = config_paths_local();
else
    error("config_paths_local.m not found. Copy configs/config_paths_template.m to configs/config_paths_local.m and edit local paths.");
end

featureFile = fullfile(paths.featureOutputDir, "AllData_FileLevelFeatures.xlsx");

outDir = fullfile(paths.outputRoot, "05_fixed_cabess_validation");

if ~exist(outDir, "dir")
    mkdir(outDir);
end

rng(42);

%% =========================================================
% 1. Analysis settings
%% =========================================================

cfg = struct();

% "progression_core": recommended default for paper table.
% "all_pairwise"     : runs all pairwise tasks used in the previous ML script.
cfg.taskPreset = "progression_core";

% Reproduce the primary task definition used in the manuscript:
% 325 maintained CN/SCD records versus 45 decline-to-dementia records.
% The audit stops execution if label construction changes unexpectedly.
cfg.auditExpectedMainCounts = true;
cfg.expectedMainClassA = 325;
cfg.expectedMainClassB = 45;

% Same nested feature-selection settings as the previous ML/fuzzy pipeline
cfg.topKList = [3 5 7 8 10 15 20 30];
cfg.rankMethods = ["absD", "aucDistance"];
cfg.transformModes = ["zscore", "log_zscore", "winsor_log_zscore"];

% CV settings
cfg.nRepeats = 30;
cfg.kFold = 5;
cfg.minClassN = 5;
cfg.randomSeed = 42;
cfg.splitMode = "SubjectLevel";

% All eligible records are retained.
% Class imbalance is handled using inverse-frequency weights estimated from
% the training fold only. Record-level random downsampling is not used.
cfg.useClassWeights = true;
cfg.requireBothClassesPerFold = true;
cfg.maxPartitionAttempts = 200;

% Feature modes
cfg.runAllFeatures = true;
cfg.runTopK = true;

% Final seven-feature signature selected by
% TopChild_ByBalancedAccuracy_Rank2 during the CaBESS discovery stage.
cfg.includeFixedCaBESS7 = true;
cfg.fixedCaBESS7 = ["ch1_theta_rel_mean", ...
                    "ch1_theta_rel_iqr", ...
                    "ch1_alpha_rel_mean", ...
                    "ch1_beta_rel_mean", ...
                    "ch1_theta_alpha_ratio_med", ...
                    "ch2_theta_alpha_ratio_std", ...
                    "ch1_sef95_std"];

% Bayesian logistic regression settings
bayesOpt = struct();
bayesOpt.priorStd = 0.5;
bayesOpt.interceptPriorStd = 20;
bayesOpt.maxIter = 200;
bayesOpt.tol = 1e-6;
bayesOpt.nPosteriorSamples = 1000;
bayesOpt.threshold = 0.5;
bayesOpt.jitter = 1e-6;

fprintf("\n=== Bayesian-only benchmark: preprocessing x feature count x class definition ===\n");
fprintf("Feature file: %s\n", featureFile);
fprintf("Output dir  : %s\n", outDir);
fprintf("Task preset : %s\n", cfg.taskPreset);
fprintf("Split mode  : %s\n", cfg.splitMode);
fprintf("Class handling: training-fold inverse-frequency weighting\n");

diary(fullfile(outDir, "FixedCaBESS_Validation_Log.txt"));

%% =========================================================
% 2. Load data and construct labels
%% =========================================================

DATA0 = readtable(featureFile, "VariableNamingRule", "preserve");

fprintf("\n=== Loaded file-level feature data ===\n");
fprintf("Rows: %d\n", height(DATA0));

requiredVars = ["Strict_Label", "Strict_Subtype", "Current_Diagnosis"];
for i = 1:numel(requiredVars)
    if ~ismember(requiredVars(i), string(DATA0.Properties.VariableNames))
        error("%s column is required.", requiredVars(i));
    end
end

DATA0.Strict_Label = string(strtrim(DATA0.Strict_Label));
DATA0.Strict_Subtype = string(strtrim(DATA0.Strict_Subtype));
DATA0.Current_Diagnosis = normalizeDx(DATA0.Current_Diagnosis);

DATA0.AnalysisSubtype = simplifyProgressionSubtype( ...
    DATA0.Strict_Label, DATA0.Strict_Subtype, DATA0.Current_Diagnosis);

fprintf("\n=== Current_Diagnosis distribution ===\n");
disp(countByString(DATA0.Current_Diagnosis, "Current_Diagnosis"));

fprintf("\n=== AnalysisSubtype distribution ===\n");
disp(countByString(DATA0.AnalysisSubtype, "AnalysisSubtype"));

%% =========================================================
% 3. Feature matrix
%% =========================================================

metaVars = [
    "FileKey"
    "Strict_Label"
    "Strict_Subtype"
    "AnalysisSubtype"
    "Current_Diagnosis"
    "BinaryLabel"
    "ObjectID"
    "NumSegments"
    "FullPath"
];

allVars = string(DATA0.Properties.VariableNames);
featureVars = setdiff(allVars, metaVars, "stable");

isNum = false(size(featureVars));
for i = 1:numel(featureVars)
    isNum(i) = isnumeric(DATA0.(featureVars(i)));
end

featureVars = featureVars(isNum);
X0 = table2array(DATA0(:, cellstr(featureVars)));
X0 = double(X0);

% Missing and non-finite values are intentionally retained here.
% Fold-specific median imputation is performed using training data only.

fprintf("\nNumber of numeric EEG features: %d\n", numel(featureVars));

missingFixed = setdiff(cfg.fixedCaBESS7, string(featureVars));
if ~isempty(missingFixed)
    error("The following fixed CaBESS features were not found: %s", ...
        strjoin(missingFixed, ", "));
end

fprintf("Fixed CaBESS 7-feature signature:\n");
disp(cfg.fixedCaBESS7(:));

%% =========================================================
% 4. Define tasks
%% =========================================================

tasks = defineTasks(cfg.taskPreset);

%% =========================================================
% 5. Run Bayesian-only benchmark
%% =========================================================

allResults = table();
allStability = table();
allSelected = table();
classCountRows = table();

for t = 1:numel(tasks)

    task = tasks(t);
    fprintf("\n\n====================================================\n");
    fprintf("Running task: %s\n", task.Name);
    fprintf("====================================================\n");

    source = string(DATA0.(task.SourceVar));
    idxA = ismember(source, task.ClassAValues);
    idxB = ismember(source, task.ClassBValues);
    keep = idxA | idxB;

    DATA = DATA0(keep, :);
    X = X0(keep, :);

    Y = strings(sum(keep), 1);
    Y(idxA(keep)) = task.ClassAName;
    Y(idxB(keep)) = task.ClassBName;

    DATA.BinaryLabel = Y;

    nA = sum(Y == task.ClassAName);
    nB = sum(Y == task.ClassBName);

    fprintf("Class A [%s] N = %d\n", task.ClassAName, nA);
    fprintf("Class B [%s] N = %d\n", task.ClassBName, nB);

    if cfg.auditExpectedMainCounts && ...
            string(task.Name) == "Subtype_Main_MaintainNormal_vs_DeclineToDEM"

        if nA ~= cfg.expectedMainClassA || nB ~= cfg.expectedMainClassB
            error(["Primary task count mismatch. Expected %d/%d records " + ...
                   "(non-decline/decline), but obtained %d/%d. " + ...
                   "Check simplifyProgressionSubtype and source labels."], ...
                  cfg.expectedMainClassA, cfg.expectedMainClassB, nA, nB);
        end

        fprintf("Primary task count audit passed: %d non-decline / %d decline records.\n", ...
            nA, nB);
    end

    countT = countByString(Y, "BinaryLabel");
    countT.Task = repmat(string(task.Name), height(countT), 1);
    classCountRows = [classCountRows; countT];

    if min(nA, nB) < cfg.minClassN
        fprintf("Skipped due to insufficient class size.\n");
        continue;
    end

    taskDir = fullfile(outDir, matlab.lang.makeValidName(string(task.Name)));
    if ~exist(taskDir, "dir")
        mkdir(taskDir);
    end
    writetable(DATA, fullfile(taskDir, "TaskData.xlsx"));

    [taskResults, taskStability, taskSelected] = runBayesianOnlyTask( ...
        X, Y, DATA, featureVars, task, cfg, bayesOpt);

    if ~isempty(taskResults)
        allResults = [allResults; taskResults];
        writetable(taskResults, fullfile(taskDir, "BayesianOnly_RawResults.xlsx"));
    end

    if ~isempty(taskStability)
        allStability = [allStability; taskStability];
        writetable(taskStability, fullfile(taskDir, "BayesianOnly_FeatureSelectionStability.xlsx"));
    end

    if ~isempty(taskSelected)
        allSelected = [allSelected; taskSelected];
        writetable(taskSelected, fullfile(taskDir, "BayesianOnly_SelectedFeatures_ByConfig.xlsx"));
    end
end

writetable(classCountRows, fullfile(outDir, "BayesianOnly_TaskClassCounts.xlsx"));
writetable(allResults, fullfile(outDir, "BayesianOnly_RawResults.xlsx"));
writetable(allStability, fullfile(outDir, "BayesianOnly_FeatureSelectionStability.xlsx"));
writetable(allSelected, fullfile(outDir, "BayesianOnly_SelectedFeatures_ByConfig.xlsx"));

summaryTable = summarizeBayesianOnlyResults(allResults);
writetable(summaryTable, fullfile(outDir, "BayesianOnly_Summary.xlsx"));

if ~isempty(summaryTable)
    bestBA = sortrows(summaryTable, ["Mean_BalancedAccuracy", "Mean_AUC"], "descend");
    writetable(bestBA, fullfile(outDir, "BayesianOnly_BestByBalancedAccuracy.xlsx"));

    fprintf("\n=== Bayesian-only top configurations ===\n");
    disp(bestBA(1:min(30,height(bestBA)), :));
end

makeSummaryPlots(summaryTable, outDir);

fprintf("\nDone. Results saved to:\n%s\n", outDir);
diary off;

%% =========================================================
% Local functions
%% =========================================================

function tasks = defineTasks(taskPreset)

    tasks = struct([]);

    tasks(end+1).Name = "Subtype_Main_MaintainNormal_vs_DeclineToDEM";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Maintain_CN", "Maintain_SCD"];
    tasks(end).ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM"];
    tasks(end).ClassAName = "Maintain_CN_SCD";
    tasks(end).ClassBName = "Decline_to_DEM";

    tasks(end+1).Name = "Subtype_Relaxed_MaintainNormal_vs_DeclineToDEM";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Maintain_CN", "Maintain_SCD"];
    tasks(end).ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM", "Decline_to_DEM_Other"];
    tasks(end).ClassAName = "Maintain_CN_SCD";
    tasks(end).ClassBName = "Decline_to_DEM";

    tasks(end+1).Name = "Subtype_MaintainNormal_vs_DeclineToMCI";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Maintain_CN", "Maintain_SCD"];
    tasks(end).ClassBValues = ["Decline_to_MCI"];
    tasks(end).ClassAName = "Maintain_CN_SCD";
    tasks(end).ClassBName = "Decline_to_MCI";

    tasks(end+1).Name = "Subtype_DeclineToMCI_vs_DeclineToDEM";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Decline_to_MCI"];
    tasks(end).ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM", "Decline_to_DEM_Other"];
    tasks(end).ClassAName = "Decline_to_MCI";
    tasks(end).ClassBName = "Decline_to_DEM";

    if string(taskPreset) ~= "all_pairwise"
        return;
    end

    tasks(end+1).Name = "Current_CN_SCD_vs_MCI";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["CN", "SCD"];
    tasks(end).ClassBValues = ["MCI"];
    tasks(end).ClassAName = "CN_SCD";
    tasks(end).ClassBName = "MCI";

    tasks(end+1).Name = "Current_CN_SCD_vs_DEM";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["CN", "SCD"];
    tasks(end).ClassBValues = ["DEM/D"];
    tasks(end).ClassAName = "CN_SCD";
    tasks(end).ClassBName = "DEM";

    tasks(end+1).Name = "Current_CN_vs_MCI";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["CN"];
    tasks(end).ClassBValues = ["MCI"];
    tasks(end).ClassAName = "CN";
    tasks(end).ClassBName = "MCI";

    tasks(end+1).Name = "Current_CN_vs_DEM";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["CN"];
    tasks(end).ClassBValues = ["DEM/D"];
    tasks(end).ClassAName = "CN";
    tasks(end).ClassBName = "DEM";

    tasks(end+1).Name = "Current_MCI_vs_DEM";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["MCI"];
    tasks(end).ClassBValues = ["DEM/D"];
    tasks(end).ClassAName = "MCI";
    tasks(end).ClassBName = "DEM";

    tasks(end+1).Name = "Current_SCD_vs_MCI";
    tasks(end).SourceVar = "Current_Diagnosis";
    tasks(end).ClassAValues = ["SCD"];
    tasks(end).ClassBValues = ["MCI"];
    tasks(end).ClassAName = "SCD";
    tasks(end).ClassBName = "MCI";

    tasks(end+1).Name = "Subtype_MaintainCN_vs_DeclineNormalToDEM";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Maintain_CN"];
    tasks(end).ClassBValues = ["Decline_Normal_to_DEM"];
    tasks(end).ClassAName = "Maintain_CN";
    tasks(end).ClassBName = "Decline_Normal_to_DEM";

    tasks(end+1).Name = "Subtype_MaintainCN_vs_DeclineMCIToDEM";
    tasks(end).SourceVar = "AnalysisSubtype";
    tasks(end).ClassAValues = ["Maintain_CN"];
    tasks(end).ClassBValues = ["Decline_MCI_to_DEM"];
    tasks(end).ClassAName = "Maintain_CN";
    tasks(end).ClassBName = "Decline_MCI_to_DEM";
end


function [taskResults, taskStability, taskSelected] = runBayesianOnlyTask( ...
    X, Y, DATA, featureVars, task, cfg, bayesOpt)

    taskResults = table();
    taskStability = table();
    taskSelected = table();

    if cfg.runAllFeatures
        for tm = 1:numel(cfg.transformModes)
            transformMode = cfg.transformModes(tm);

            fprintf("\nRunning Bayesian: Task=%s | Transform=%s | FeatureSet=AllFeatures\n", ...
                task.Name, transformMode);

            [resT, ~, selT] = runOneConfigRepeatedCV( ...
                X, Y, DATA, featureVars, task, cfg, bayesOpt, ...
                transformMode, "none", "AllFeatures", numel(featureVars));

            taskResults = [taskResults; resT];
            taskSelected = [taskSelected; selT];
        end
    end

    if cfg.runTopK
        for tm = 1:numel(cfg.transformModes)
            transformMode = cfg.transformModes(tm);

            for rm = 1:numel(cfg.rankMethods)
                rankMethod = cfg.rankMethods(rm);

                for kk = 1:numel(cfg.topKList)
                    topK = cfg.topKList(kk);

                    fprintf("\nRunning Bayesian: Task=%s | Transform=%s | Rank=%s | TopK=%d\n", ...
                        task.Name, transformMode, rankMethod, topK);

                    [resT, stabT, selT] = runOneConfigRepeatedCV( ...
                        X, Y, DATA, featureVars, task, cfg, bayesOpt, ...
                        transformMode, rankMethod, "TopK", topK);

                    taskResults = [taskResults; resT];
                    taskStability = [taskStability; stabT];
                    taskSelected = [taskSelected; selT];
                end
            end
        end
    end

    if cfg.includeFixedCaBESS7
        [isMember, fixedIdx] = ismember(cfg.fixedCaBESS7, string(featureVars));
        fixedIdx = fixedIdx(isMember);

        if ~isempty(fixedIdx)
            for tm = 1:numel(cfg.transformModes)
                transformMode = cfg.transformModes(tm);

                fprintf("\nRunning Bayesian: Task=%s | Transform=%s | FeatureSet=Fixed_CaBESS7\n", ...
                    task.Name, transformMode);

                [resT, ~, selT] = runOneConfigRepeatedCV( ...
                    X, Y, DATA, featureVars, task, cfg, bayesOpt, ...
                    transformMode, "fixed", "Fixed_CaBESS7", numel(fixedIdx), fixedIdx);

                taskResults = [taskResults; resT];
                taskSelected = [taskSelected; selT];
            end
        else
            warning("Fixed CaBESS7 features were not found in featureVars. Fixed mode skipped.");
        end
    end
end


function [configResults, stabilityT, selectedT] = runOneConfigRepeatedCV( ...
    X, Y, DATA, featureVars, task, cfg, bayesOpt, ...
    transformMode, rankMethod, featureSetName, topK, fixedIdx)

    if nargin < 12
        fixedIdx = [];
    end

    configResults = table();
    stabilityT = table();
    selectedT = table();

    selectedCount = zeros(numel(featureVars), 1);
    totalFolds = 0;

    classAName = string(task.ClassAName);
    classBName = string(task.ClassBName);
    Yr = string(Y(:));
    yBin = double(Yr == classBName);

    [nSubjectsTotal, nSubjectsA, nSubjectsB, nMixedSubjects] = ...
        countSubjectsByClass(DATA, Yr, classAName, classBName);

    for rep = 1:cfg.nRepeats

        partitionSeed = cfg.randomSeed + rep - 1;

        [foldId, thisKFold, partitionInfo] = makeSubjectLevelFolds( ...
            DATA, Yr, classAName, classBName, cfg.kFold, partitionSeed, ...
            cfg.requireBothClassesPerFold, cfg.maxPartitionAttempts);

        allTrue = strings(0, 1);
        allPred = strings(0, 1);
        allProb = [];
        allProbStd = [];
        allEntropy = [];
        allCIWidth = [];

        selectedFoldStrings = strings(0, 1);

        for fold = 1:thisKFold

            te = foldId == fold;
            tr = ~te;

            if ~any(tr) || ~any(te)
                warning("Repeat %d fold %d contains an empty train or test set. Fold skipped.", rep, fold);
                continue;
            end

            Ytr = Yr(tr);
            Yte = Yr(te);
            ytrBin = double(Ytr == classBName);

            if numel(unique(ytrBin)) < 2 || numel(unique(double(Yte == classBName))) < 2
                warning("Repeat %d fold %d does not contain both classes. Fold skipped.", rep, fold);
                continue;
            end

            Xtr = X(tr, :);
            Xte = X(te, :);

            % Training-fold-only imputation, transformation and scaling
            [XtrP, XteP] = preprocessFold(Xtr, Xte, featureVars, transformMode);

            if string(featureSetName) == "AllFeatures"
                selIdx = 1:numel(featureVars);

            elseif string(featureSetName) == "Fixed_CaBESS7"
                selIdx = fixedIdx;

            else
                rankingScore = rankFeatures( ...
                    XtrP, Ytr, classAName, classBName, rankMethod);

                [~, ord] = sort( ...
                    rankingScore, "descend", "MissingPlacement", "last");

                useK = min(topK, numel(ord));
                selIdx = sort(ord(1:useK));
            end

            if isempty(selIdx)
                error("No features were selected for %s.", featureSetName);
            end

            selectedCount(selIdx) = selectedCount(selIdx) + 1;
            totalFolds = totalFolds + 1;
            selectedFoldStrings(end+1, 1) = ...
                strjoin(string(featureVars(selIdx)), "; "); %#ok<AGROW>

            XtrSel = XtrP(:, selIdx);
            XteSel = XteP(:, selIdx);

            sampleWeights = makeClassWeights(ytrBin, cfg.useClassWeights);

            mdl = trainBayesianLogisticLaplaceWeighted( ...
                XtrSel, ytrBin, sampleWeights, bayesOpt);

            pred = predictBayesianLogistic( ...
                mdl, XteSel, bayesOpt.nPosteriorSamples);

            yhatBin = double(pred.probMean >= bayesOpt.threshold);

            Ypred = strings(numel(yhatBin), 1);
            Ypred(yhatBin == 0) = classAName;
            Ypred(yhatBin == 1) = classBName;

            allTrue = [allTrue; Yte(:)];
            allPred = [allPred; Ypred(:)];
            allProb = [allProb; pred.probMean(:)];
            allProbStd = [allProbStd; pred.probStd(:)];
            allEntropy = [allEntropy; pred.entropy(:)];
            allCIWidth = [allCIWidth; ...
                pred.probCIUpper(:) - pred.probCILower(:)];
        end

        if isempty(allTrue)
            warning("Repeat %d produced no valid test predictions and was skipped.", rep);
            continue;
        end

        metric = computeMetricsString( ...
            allTrue, allPred, allProb, allProbStd, allEntropy, allCIWidth, ...
            classAName, classBName);

        newRow = table();
        newRow.Task = string(task.Name);
        newRow.ClassA = classAName;
        newRow.ClassB = classBName;
        newRow.N_ClassA_Total = sum(Yr == classAName);
        newRow.N_ClassB_Total = sum(Yr == classBName);
        newRow.N_Subjects_Total = nSubjectsTotal;
        newRow.N_Subjects_ClassA = nSubjectsA;
        newRow.N_Subjects_ClassB = nSubjectsB;
        newRow.N_MixedLabel_Subjects = nMixedSubjects;
        newRow.SplitMode = string(cfg.splitMode);
        newRow.ClassHandling = "TrainingFoldClassWeights";
        newRow.KFold = thisKFold;
        newRow.PartitionAttempts = partitionInfo.Attempts;
        newRow.TransformMode = string(transformMode);
        newRow.RankMethod = string(rankMethod);
        newRow.FeatureSet = string(featureSetName);
        newRow.TopK = topK;
        newRow.Model = "BayesianLogistic_Laplace";
        newRow.Repeat = rep;
        newRow.Accuracy = metric.Accuracy;
        newRow.BalancedAccuracy = metric.BalancedAccuracy;
        newRow.MacroF1 = metric.MacroF1;
        newRow.Recall_ClassA = metric.Recall_ClassA;
        newRow.Recall_ClassB = metric.Recall_ClassB;
        newRow.Precision_ClassA = metric.Precision_ClassA;
        newRow.Precision_ClassB = metric.Precision_ClassB;
        newRow.AUC = metric.AUC;
        newRow.BrierScore = metric.BrierScore;
        newRow.NLL = metric.NLL;
        newRow.ECE = metric.ECE;
        newRow.MeanProbStd = metric.MeanProbStd;
        newRow.MeanEntropy = metric.MeanEntropy;
        newRow.MeanCIWidth = metric.MeanCIWidth;

        configResults = [configResults; newRow];

        selRow = table();
        selRow.Task = string(task.Name);
        selRow.SplitMode = string(cfg.splitMode);
        selRow.TransformMode = string(transformMode);
        selRow.RankMethod = string(rankMethod);
        selRow.FeatureSet = string(featureSetName);
        selRow.TopK = topK;
        selRow.Repeat = rep;
        selRow.SelectedFeatures_ConcatenatedAcrossFolds = ...
            strjoin(selectedFoldStrings, " || ");
        selectedT = [selectedT; selRow];
    end

    if totalFolds > 0
        stabilityT = table();
        stabilityT.Task = repmat(string(task.Name), numel(featureVars), 1);
        stabilityT.SplitMode = repmat(string(cfg.splitMode), numel(featureVars), 1);
        stabilityT.TransformMode = repmat(string(transformMode), numel(featureVars), 1);
        stabilityT.RankMethod = repmat(string(rankMethod), numel(featureVars), 1);
        stabilityT.FeatureSet = repmat(string(featureSetName), numel(featureVars), 1);
        stabilityT.TopK = repmat(topK, numel(featureVars), 1);
        stabilityT.Model = repmat("BayesianLogistic_Laplace", numel(featureVars), 1);
        stabilityT.Feature = featureVars(:);
        stabilityT.SelectionCount = selectedCount;
        stabilityT.SelectionFrequency = selectedCount ./ totalFolds;

        stabilityT = sortrows(stabilityT, ...
            ["Task", "SplitMode", "TransformMode", "RankMethod", ...
             "FeatureSet", "TopK", "SelectionFrequency"], ...
            ["ascend", "ascend", "ascend", "ascend", ...
             "ascend", "ascend", "descend"]);
    end
end


function [foldId, kFoldUsed, info] = makeSubjectLevelFolds( ...
    DATA, Y, classAName, classBName, requestedK, seed, ...
    requireBothClasses, maxAttempts)

    vars = string(DATA.Properties.VariableNames);
    if ~ismember("ObjectID", vars)
        error("ObjectID is required for subject-level cross-validation.");
    end

    subjectID = strtrim(string(DATA.ObjectID));

    % A missing ObjectID is treated as a unique record-specific identifier
    % so that unrelated missing IDs are not grouped together.
    missingID = ismissing(subjectID) | subjectID == "";
    missingRows = find(missingID);
    for i = 1:numel(missingRows)
        subjectID(missingRows(i)) = ...
            "__MissingObjectID_Row_" + string(missingRows(i));
    end

    [uniqueSubjects, ~, subjectGroup] = unique(subjectID, "stable");
    nSubjects = numel(uniqueSubjects);

    subjectLabel = zeros(nSubjects, 1);
    mixedSubject = false(nSubjects, 1);

    for s = 1:nSubjects
        ys = string(Y(subjectGroup == s));
        hasA = any(ys == classAName);
        hasB = any(ys == classBName);

        mixedSubject(s) = hasA && hasB;

        % For stratification only, any decline record assigns the subject
        % to the decline stratum. Original record labels remain unchanged.
        subjectLabel(s) = double(hasB);
    end

    if any(mixedSubject)
        warning("%d ObjectID values contain records from both task classes. " + ...
            "They remain grouped in one fold and are stratified as decline subjects.", ...
            sum(mixedSubject));
    end

    nSubjectA = sum(subjectLabel == 0);
    nSubjectB = sum(subjectLabel == 1);
    kFoldUsed = min([requestedK, nSubjectA, nSubjectB]);

    if kFoldUsed < 2
        error("Insufficient subject-level class counts for cross-validation: " + ...
            "ClassA subjects=%d, ClassB subjects=%d.", nSubjectA, nSubjectB);
    end

    success = false;
    foldId = zeros(height(DATA), 1);

    for attempt = 1:maxAttempts
        rng(seed + attempt - 1, "twister");

        cv = cvpartition(categorical(subjectLabel), "KFold", kFoldUsed);
        candidateFoldId = zeros(height(DATA), 1);

        for fold = 1:kFoldUsed
            testSubjectIdx = find(test(cv, fold));
            candidateFoldId(ismember(subjectGroup, testSubjectIdx)) = fold;
        end

        valid = all(candidateFoldId > 0);

        if valid && requireBothClasses
            for fold = 1:kFoldUsed
                te = candidateFoldId == fold;
                tr = ~te;

                yTe = double(string(Y(te)) == classBName);
                yTr = double(string(Y(tr)) == classBName);

                if numel(unique(yTe)) < 2 || numel(unique(yTr)) < 2
                    valid = false;
                    break;
                end
            end
        end

        if valid
            foldId = candidateFoldId;
            success = true;
            break;
        end
    end

    if ~success
        error("Failed to generate a valid subject-level partition after %d attempts.", ...
            maxAttempts);
    end

    info = struct();
    info.Attempts = attempt;
    info.NSubjects = nSubjects;
    info.NSubjectsClassA = nSubjectA;
    info.NSubjectsClassB = nSubjectB;
    info.NMixedLabelSubjects = sum(mixedSubject);
end


function [nTotal, nA, nB, nMixed] = countSubjectsByClass( ...
    DATA, Y, classAName, classBName)

    vars = string(DATA.Properties.VariableNames);
    if ~ismember("ObjectID", vars)
        error("ObjectID is required for subject-level summaries.");
    end

    subjectID = strtrim(string(DATA.ObjectID));
    missingID = ismissing(subjectID) | subjectID == "";
    missingRows = find(missingID);

    for i = 1:numel(missingRows)
        subjectID(missingRows(i)) = ...
            "__MissingObjectID_Row_" + string(missingRows(i));
    end

    [uniqueSubjects, ~, subjectGroup] = unique(subjectID, "stable");

    nTotal = numel(uniqueSubjects);
    nA = 0;
    nB = 0;
    nMixed = 0;

    for s = 1:nTotal
        ys = string(Y(subjectGroup == s));
        hasA = any(ys == classAName);
        hasB = any(ys == classBName);

        if hasA && hasB
            nMixed = nMixed + 1;
        end

        if hasB
            nB = nB + 1;
        elseif hasA
            nA = nA + 1;
        end
    end
end


function [XtrP, XteP] = preprocessFold(Xtr, Xte, featureVars, transformMode)

    XtrP = double(Xtr);
    XteP = double(Xte);

    % Training-fold-only median imputation
    for j = 1:size(XtrP, 2)
        trainCol = XtrP(:, j);
        medVal = median(trainCol(isfinite(trainCol)), "omitnan");

        if isempty(medVal) || ~isfinite(medVal)
            medVal = 0;
        end

        XtrP(~isfinite(XtrP(:, j)), j) = medVal;
        XteP(~isfinite(XteP(:, j)), j) = medVal;
    end

    if transformMode == "log_zscore" || transformMode == "winsor_log_zscore"

        for j = 1:numel(featureVars)

            f = featureVars(j);

            isCorrCentral = contains(f, "corr_mean") || contains(f, "corr_med");

            isLogCandidate = contains(f, "theta_rel") || ...
                             contains(f, "alpha_rel") || ...
                             contains(f, "beta_rel") || ...
                             contains(f, "delta_rel") || ...
                             contains(f, "gamma_rel") || ...
                             contains(f, "theta_alpha_ratio") || ...
                             contains(f, "median_freq") || ...
                             contains(f, "sef95") || ...
                             contains(f, "_std") || ...
                             contains(f, "_iqr");

            if isCorrCentral
                XtrP(:, j) = atanh(max(min(XtrP(:, j), 0.999), -0.999));
                XteP(:, j) = atanh(max(min(XteP(:, j), 0.999), -0.999));

            elseif isLogCandidate && min(XtrP(:, j), [], "omitnan") >= 0
                XtrP(:, j) = log(XtrP(:, j) + eps);
                XteP(:, j) = log(max(XteP(:, j), 0) + eps);
            end
        end
    end

    if transformMode == "winsor_log_zscore"
        lo = prctile(XtrP, 1, 1);
        hi = prctile(XtrP, 99, 1);

        for j = 1:size(XtrP, 2)
            XtrP(:, j) = min(max(XtrP(:, j), lo(j)), hi(j));
            XteP(:, j) = min(max(XteP(:, j), lo(j)), hi(j));
        end
    end

    mu = mean(XtrP, 1, "omitnan");
    sig = std(XtrP, 0, 1, "omitnan");
    sig(sig < eps | ~isfinite(sig)) = 1;
    mu(~isfinite(mu)) = 0;

    XtrP = (XtrP - mu) ./ sig;
    XteP = (XteP - mu) ./ sig;

    XtrP(~isfinite(XtrP)) = 0;
    XteP(~isfinite(XteP)) = 0;
end


function score = rankFeatures(Xtr, Ytr, classAName, classBName, rankMethod)

    Ytr = string(Ytr);
    nF = size(Xtr, 2);
    score = nan(nF, 1);

    for j = 1:nF

        x = Xtr(:, j);

        switch string(rankMethod)

            case "absD"
                d = cohensDTwoClass(x, Ytr, classAName, classBName);
                score(j) = abs(d);

            case "aucDistance"
                aucVal = simpleAUC(x, Ytr, classBName);
                score(j) = abs(aucVal - 0.5);

            otherwise
                d = cohensDTwoClass(x, Ytr, classAName, classBName);
                score(j) = abs(d);
        end
    end

    score(~isfinite(score)) = 0;
end


function d = cohensDTwoClass(x, y, classA, classB)

    x = double(x);
    y = string(y);

    xa = x(y == classA);
    xb = x(y == classB);

    xa = xa(isfinite(xa));
    xb = xb(isfinite(xb));

    if numel(xa) < 2 || numel(xb) < 2
        d = nan;
        return;
    end

    pooledStd = sqrt(((numel(xa)-1)*var(xa) + (numel(xb)-1)*var(xb)) / ...
                     (numel(xa)+numel(xb)-2));

    d = (mean(xb) - mean(xa)) / (pooledStd + eps);
end


function w = makeClassWeights(yBin, useWeights)

    yBin = double(yBin(:));
    w = ones(size(yBin));

    if ~useWeights
        return;
    end

    n = numel(yBin);
    n0 = sum(yBin == 0);
    n1 = sum(yBin == 1);

    if n0 > 0
        w(yBin == 0) = n / (2*n0);
    end

    if n1 > 0
        w(yBin == 1) = n / (2*n1);
    end
end


function mdl = trainBayesianLogisticLaplaceWeighted(X, y, sampleWeights, opt)

    y = double(y(:));
    sampleWeights = double(sampleWeights(:));

    n = size(X, 1);
    p = size(X, 2);

    Xb = [ones(n, 1), X];

    beta = zeros(p + 1, 1);

    priorPrec = zeros(p + 1, 1);
    priorPrec(1) = 1 / (opt.interceptPriorStd^2);
    priorPrec(2:end) = 1 / (opt.priorStd^2);

    for iter = 1:opt.maxIter

        eta = Xb * beta;
        prob = sigmoidStable(eta);

        W = sampleWeights .* prob .* (1 - prob);
        W = max(W, 1e-8);

        grad = Xb' * (sampleWeights .* (prob - y)) + priorPrec .* beta;

        H = Xb' * (Xb .* W) + diag(priorPrec);
        H = H + opt.jitter * eye(p + 1);

        step = H \ grad;
        betaNew = beta - step;

        if norm(betaNew - beta) < opt.tol * (1 + norm(beta))
            beta = betaNew;
            break;
        end

        beta = betaNew;
    end

    eta = Xb * beta;
    prob = sigmoidStable(eta);
    W = sampleWeights .* prob .* (1 - prob);
    W = max(W, 1e-8);

    H = Xb' * (Xb .* W) + diag(priorPrec);
    H = H + opt.jitter * eye(p + 1);

    covBeta = safeInverseSPD(H, opt.jitter);

    mdl = struct();
    mdl.betaMAP = beta;
    mdl.covBeta = covBeta;
    mdl.priorStd = opt.priorStd;
    mdl.interceptPriorStd = opt.interceptPriorStd;
    mdl.nIter = iter;
end


function pred = predictBayesianLogistic(mdl, X, nSamples)

    n = size(X, 1);
    Xb = [ones(n, 1), X];

    betaSamples = sampleMultivariateNormal(mdl.betaMAP, mdl.covBeta, nSamples);

    probSamples = sigmoidStable(Xb * betaSamples');

    probMean = mean(probSamples, 2);
    probStd = std(probSamples, 0, 2);

    probCILower = quantile(probSamples, 0.025, 2);
    probCIUpper = quantile(probSamples, 0.975, 2);

    entropy = binaryEntropy(probMean);

    pred = struct();
    pred.probMean = probMean;
    pred.probStd = probStd;
    pred.probCILower = probCILower;
    pred.probCIUpper = probCIUpper;
    pred.entropy = entropy;
end


function betaSamples = sampleMultivariateNormal(mu, Sigma, nSamples)

    mu = mu(:);
    d = numel(mu);

    Sigma = (Sigma + Sigma') / 2;

    jitter = 1e-8;
    success = false;

    for k = 1:8
        [L, flag] = chol(Sigma + jitter * eye(d), "lower");
        if flag == 0
            success = true;
            break;
        end
        jitter = jitter * 10;
    end

    if ~success
        [V, D] = eig(Sigma);
        eigVals = max(diag(D), 1e-10);
        L = V * diag(sqrt(eigVals));
    end

    betaSamples = mu' + randn(nSamples, d) * L';
end


function invA = safeInverseSPD(A, jitter)

    A = (A + A') / 2;
    d = size(A, 1);

    for k = 1:8
        [L, flag] = chol(A + jitter * eye(d), "lower");
        if flag == 0
            invA = L' \ (L \ eye(d));
            invA = (invA + invA') / 2;
            return;
        end
        jitter = jitter * 10;
    end

    invA = pinv(A);
    invA = (invA + invA') / 2;
end


function s = sigmoidStable(z)

    s = zeros(size(z));

    idx = z >= 0;
    s(idx) = 1 ./ (1 + exp(-z(idx)));

    ez = exp(z(~idx));
    s(~idx) = ez ./ (1 + ez);

    s = min(max(s, 1e-12), 1 - 1e-12);
end


function h = binaryEntropy(p)

    p = min(max(p, 1e-12), 1 - 1e-12);
    h = -(p .* log(p) + (1 - p) .* log(1 - p));
end


function metric = computeMetricsString( ...
    yTrue, yPred, probB, probStd, entropy, ciWidth, classAName, classBName)

    yTrue = string(yTrue(:));
    yPred = string(yPred(:));
    probB = double(probB(:));

    yBin = double(yTrue == classBName);

    acc = mean(yTrue == yPred);

    TP_A = sum(yTrue == classAName & yPred == classAName);
    FP_A = sum(yTrue ~= classAName & yPred == classAName);
    FN_A = sum(yTrue == classAName & yPred ~= classAName);

    TP_B = sum(yTrue == classBName & yPred == classBName);
    FP_B = sum(yTrue ~= classBName & yPred == classBName);
    FN_B = sum(yTrue == classBName & yPred ~= classBName);

    precisionA = TP_A / max(TP_A + FP_A, 1);
    precisionB = TP_B / max(TP_B + FP_B, 1);

    recallA = TP_A / max(TP_A + FN_A, 1);
    recallB = TP_B / max(TP_B + FN_B, 1);

    f1A = 2 * precisionA * recallA / max(precisionA + recallA, eps);
    f1B = 2 * precisionB * recallB / max(precisionB + recallB, eps);

    p = min(max(probB, 1e-12), 1 - 1e-12);

    metric = struct();
    metric.Accuracy = acc;
    metric.BalancedAccuracy = mean([recallA, recallB], "omitnan");
    metric.MacroF1 = mean([f1A, f1B], "omitnan");
    metric.Recall_ClassA = recallA;
    metric.Recall_ClassB = recallB;
    metric.Precision_ClassA = precisionA;
    metric.Precision_ClassB = precisionB;
    metric.AUC = simpleAUC(probB, yTrue, classBName);
    metric.BrierScore = mean((probB - yBin).^2, "omitnan");
    metric.NLL = -mean(yBin .* log(p) + (1 - yBin) .* log(1 - p), "omitnan");
    metric.ECE = computeECE(probB, yBin, 10);
    metric.MeanProbStd = mean(probStd, "omitnan");
    metric.MeanEntropy = mean(entropy, "omitnan");
    metric.MeanCIWidth = mean(ciWidth, "omitnan");
end


function aucVal = simpleAUC(score, y, positiveClass)

    score = double(score(:));
    y = string(y(:));

    valid = isfinite(score);
    score = score(valid);
    y = y(valid);

    pos = y == positiveClass;
    neg = y ~= positiveClass;

    if sum(pos) < 1 || sum(neg) < 1
        aucVal = nan;
        return;
    end

    r = tiedrank(score);
    nPos = sum(pos);
    nNeg = sum(neg);

    aucVal = (sum(r(pos)) - nPos*(nPos+1)/2) / (nPos*nNeg + eps);
end


function ece = computeECE(prob, yBin, nBins)

    prob = double(prob(:));
    yBin = double(yBin(:));

    edges = linspace(0, 1, nBins + 1);
    ece = 0;
    n = numel(yBin);

    for b = 1:nBins
        if b < nBins
            idx = prob >= edges(b) & prob < edges(b + 1);
        else
            idx = prob >= edges(b) & prob <= edges(b + 1);
        end

        if ~any(idx)
            continue;
        end

        conf = mean(prob(idx));
        obs = mean(yBin(idx));

        ece = ece + sum(idx) / max(n,1) * abs(obs - conf);
    end
end


function summaryTable = summarizeBayesianOnlyResults(allResults)

    if isempty(allResults)
        summaryTable = table();
        return;
    end

    [G, task, clsA, clsB, nA, nB, splitMode, classHandling, ...
        transformMode, rankMethod, featureSet, topK, model] = findgroups( ...
        allResults.Task, ...
        allResults.ClassA, ...
        allResults.ClassB, ...
        allResults.N_ClassA_Total, ...
        allResults.N_ClassB_Total, ...
        allResults.SplitMode, ...
        allResults.ClassHandling, ...
        allResults.TransformMode, ...
        allResults.RankMethod, ...
        allResults.FeatureSet, ...
        allResults.TopK, ...
        allResults.Model);

    summaryTable = table();
    summaryTable.Task = task;
    summaryTable.ClassA = clsA;
    summaryTable.ClassB = clsB;
    summaryTable.N_ClassA_Total = nA;
    summaryTable.N_ClassB_Total = nB;
    summaryTable.SplitMode = splitMode;
    summaryTable.ClassHandling = classHandling;
    summaryTable.TransformMode = transformMode;
    summaryTable.RankMethod = rankMethod;
    summaryTable.FeatureSet = featureSet;
    summaryTable.TopK = topK;
    summaryTable.Model = model;

    metricNames = ["Accuracy", "BalancedAccuracy", "MacroF1", ...
                   "Recall_ClassA", "Recall_ClassB", ...
                   "Precision_ClassA", "Precision_ClassB", ...
                   "AUC", "BrierScore", "NLL", "ECE", ...
                   "MeanProbStd", "MeanEntropy", "MeanCIWidth"];

    for m = 1:numel(metricNames)
        met = metricNames(m);
        vals = allResults.(met);

        summaryTable.("Mean_" + met) = splitapply(@(x) mean(x, "omitnan"), vals, G);
        summaryTable.("Std_" + met) = splitapply(@(x) std(x, "omitnan"), vals, G);
    end

    summaryTable.GroupCount = splitapply(@numel, allResults.Accuracy, G);
end


function makeSummaryPlots(summaryTable, outDir)

    if isempty(summaryTable)
        return;
    end

    figDir = fullfile(outDir, "Figures");
    if ~exist(figDir, "dir")
        mkdir(figDir);
    end

    try
        sorted = sortrows(summaryTable, ["Mean_BalancedAccuracy", "Mean_AUC"], "descend");
        topN = min(25, height(sorted));
        topT = sorted(1:topN, :);

        label = topT.Task + " | " + topT.TransformMode + " | " + ...
                topT.FeatureSet + "_" + string(topT.TopK);

        fig = figure("Visible", "off", "Position", [100 100 1300 650]);
        bar(topT.Mean_BalancedAccuracy);
        hold on;
        errorbar(1:topN, topT.Mean_BalancedAccuracy, topT.Std_BalancedAccuracy, ...
            ".", "LineWidth", 1);
        ylabel("Balanced Accuracy");
        title("Bayesian-only top configurations");
        xticks(1:topN);
        xticklabels(label);
        xtickangle(45);
        grid on;
        saveas(fig, fullfile(figDir, "Fig01_BayesianOnly_TopBalancedAccuracy.png"));
        close(fig);
    catch ME
        warning("Failed to create summary plot: %s", ME.message);
    end
end


function dx = normalizeDx(x)

    x = upper(strtrim(string(x)));
    dx = strings(size(x));

    for i = 1:numel(x)

        xi = x(i);

        if xi == "" || ismissing(xi)
            dx(i) = "Unknown";
        elseif contains(xi, "진단중") || contains(xi, "UNKNOWN") || contains(xi, "NOS")
            dx(i) = "Unknown";
        elseif xi == "CN"
            dx(i) = "CN";
        elseif xi == "SCD"
            dx(i) = "SCD";
        elseif contains(xi, "MCI")
            dx(i) = "MCI";
        elseif xi == "AD" || xi == "D" || xi == "DEM" || ...
               contains(xi, "DEM") || contains(xi, "DEMENTIA") || contains(xi, "AD/D")
            dx(i) = "DEM/D";
        else
            dx(i) = "Unknown";
        end
    end
end


function subtype = simplifyProgressionSubtype(strictLabel, strictSubtype, currentDx)

    strictLabel = string(strictLabel);
    strictSubtype = string(strictSubtype);
    currentDx = string(currentDx);

    subtype = strings(size(strictLabel));

    for i = 1:numel(strictLabel)

        lab = strictLabel(i);
        sub = strictSubtype(i);
        cur = currentDx(i);

        if lab == "Maintain"

            if cur == "CN"
                subtype(i) = "Maintain_CN";
            elseif cur == "SCD"
                subtype(i) = "Maintain_SCD";
            elseif cur == "MCI"
                subtype(i) = "Maintain_MCI";
            elseif cur == "DEM/D"
                subtype(i) = "Maintain_DEM";
            else
                subtype(i) = "Maintain_Other";
            end

        elseif lab == "Declining"

            % IMPORTANT:
            % This priority exactly reproduces the label construction used
            % in the final manuscript analysis. When a compound trajectory contains both an early
            % CN/SCD-to-MCI transition and a later MCI-to-DEM transition,
            % it is assigned to Decline_to_MCI because that transition is
            % checked first. Consequently, those records are not included
            % in the primary Decline_to_DEM task, yielding the manuscript
            % count of 325 non-decline and 45 decline records.
            if contains(sub, "CN_to_MCI") || contains(sub, "SCD_to_MCI")
                subtype(i) = "Decline_to_MCI";

            elseif contains(sub, "CN_to_DEM") || contains(sub, "SCD_to_DEM")
                subtype(i) = "Decline_Normal_to_DEM";

            elseif contains(sub, "MCI_to_DEM")
                subtype(i) = "Decline_MCI_to_DEM";

            else
                if cur == "MCI"
                    subtype(i) = "Decline_to_MCI";
                elseif cur == "DEM/D"
                    subtype(i) = "Decline_to_DEM_Other";
                else
                    subtype(i) = "Decline_Other";
                end
            end

        else
            subtype(i) = "Other";
        end
    end
end


function summaryTable = countByString(x, varName)

    x = string(x);

    if isempty(x)
        summaryTable = table(strings(0,1), zeros(0,1), ...
            'VariableNames', {char(varName), 'Count'});
        return;
    end

    [G, names] = findgroups(x);
    counts = splitapply(@numel, x, G);

    summaryTable = table(names, counts, ...
        'VariableNames', {char(varName), 'Count'});
end
