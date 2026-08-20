%% 06_run_bopa_analysis.m
% BOPA: posterior feature contribution and one-feature output-preserving analysis.
% Bayesian output-preserving attribution surface analysis
%
% This script visualizes the trained Bayesian logistic model at the
% explanation level. It shows that feature contribution is not a single
% value but a posterior distribution because beta is a posterior distribution.
%
% Main outputs:
% 1) Feature contribution distribution for representative cases
% 2) One-feature output-preserving sweep
% 3) Output-preserving range per feature
% 4) Contribution surface and predictive uncertainty surface
%
% Default final 7-feature signature:
%   ch1_theta_rel_mean
%   ch1_theta_rel_iqr
%   ch1_alpha_rel_mean
%   ch1_beta_rel_mean
%   ch1_theta_alpha_ratio_med
%   ch2_theta_alpha_ratio_std
%   ch1_sef95_std
%
% Important:
% This script trains one final Bayesian model using all target rows for
% model visualization. Use 05_validate_fixed_cabess_signature.m for repeated-CV performance.

clc; clear; close all;

%% =========================================================
% 0. Settings
%% =========================================================

selectedFeatureMode = "Manual";  % "Manual" or "FromValidation"
signatureName = "TopChild_ByBalancedAccuracy_Rank2";

manualSelectedFeatures = [
    "ch1_theta_rel_mean"
    "ch1_theta_rel_iqr"
    "ch1_alpha_rel_mean"
    "ch1_beta_rel_mean"
    "ch1_theta_alpha_ratio_med"
    "ch2_theta_alpha_ratio_std"
    "ch1_sef95_std"
];

bayesOpt = struct();
bayesOpt.priorStd = 0.5;
bayesOpt.interceptPriorStd = 20;
bayesOpt.maxIter = 200;
bayesOpt.tol = 1e-6;
bayesOpt.jitter = 1e-6;

nPosteriorSamples = 6000;
randomSeed = 20260704;

% One-feature perturbation grid in z-score units.
sweepOffsetZ = linspace(-2.5, 2.5, 101);
clipSweepToObservedRange = true;
observedPrctileRange = [1, 99];

% Output-preserving definition.
minPosteriorAgreement = 0.80;
useConfidencePreserving = false;
maxProbMeanChange = 0.10;

representativeModes = [
    "ConfidentMaintain"
    "ConfidentDecline"
    "Boundary"
    "HighUncertainty"
];
manualTaskRowIndices = [];  % e.g., [10 24 57]

% Target label used in the current main study.
task.Name = "Subtype_MaintainNormal_vs_DeclineToDEM";
task.SourceVar = "AnalysisSubtype";
task.ClassAValues = ["Maintain_CN", "Maintain_SCD"];
task.ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM"];
task.ClassAName = "Maintain_CN_SCD";
task.ClassBName = "Decline_to_DEM";

%% =========================================================
% 1. Path setting
%% =========================================================

projectRoot = fileparts(mfilename("fullpath"));

if exist(fullfile(projectRoot, "configs"), "dir")
    addpath(genpath(fullfile(projectRoot, "configs")));
end
if exist(fullfile(projectRoot, "src"), "dir")
    addpath(genpath(fullfile(projectRoot, "src")));
end

if exist("config_paths_local", "file") ~= 2
    error(["config_paths_local.m not found. Copy configs/config_paths_template.m " + ...
           "to configs/config_paths_local.m and edit local paths."]);
end
paths = config_paths_local();
featureFile = fullfile(paths.featureOutputDir, "AllData_FileLevelFeatures.xlsx");
validationOutDir = fullfile(paths.outputRoot, "05_fixed_cabess_validation");
outDir = fullfile(paths.outputRoot, "06_bopa");

figDir = fullfile(outDir, "Figures");
if ~exist(outDir, "dir"); mkdir(outDir); end
if ~exist(figDir, "dir"); mkdir(figDir); end

fprintf("\n=== Bayesian output-preserving attribution surface analysis ===\n");
fprintf("Feature file: %s\n", featureFile);
fprintf("Output dir  : %s\n", outDir);
assert(exist(featureFile, "file") == 2, "Feature file not found: %s", featureFile);

%% =========================================================
% 2. Load data and construct task
%% =========================================================

DATA = readtable(featureFile, "VariableNamingRule", "preserve");

requiredVars = ["Strict_Label", "Strict_Subtype", "Current_Diagnosis"];
for i = 1:numel(requiredVars)
    assert(ismember(requiredVars(i), string(DATA.Properties.VariableNames)), ...
        "%s column is required.", requiredVars(i));
end

DATA.Strict_Label = string(strtrim(DATA.Strict_Label));
DATA.Strict_Subtype = string(strtrim(DATA.Strict_Subtype));
DATA.Current_Diagnosis = normalizeDx(DATA.Current_Diagnosis);
DATA.AnalysisSubtype = simplifyProgressionSubtype( ...
    DATA.Strict_Label, DATA.Strict_Subtype, DATA.Current_Diagnosis);

metaVars = [
    "FileKey"
    "Strict_Label"
    "Strict_Subtype"
    "AnalysisSubtype"
    "Current_Diagnosis"
    "ObjectID"
    "NumSegments"
    "FullPath"
];

allVars = string(DATA.Properties.VariableNames);
featureVars = setdiff(allVars, metaVars, "stable");

isNum = false(size(featureVars));
for i = 1:numel(featureVars)
    isNum(i) = isnumeric(DATA.(featureVars(i)));
end
featureVars = featureVars(isNum);

Xall = table2array(DATA(:, cellstr(featureVars)));
Xall = replaceInvalidWithMedian(Xall);

source = string(DATA.(task.SourceVar));
idxA = ismember(source, task.ClassAValues);
idxB = ismember(source, task.ClassBValues);
keep = idxA | idxB;

DATA_task = DATA(keep, :);
X_task_allFeatures = Xall(keep, :);

Y = strings(sum(keep), 1);
Y(idxA(keep)) = task.ClassAName;
Y(idxB(keep)) = task.ClassBName;
yBin = double(Y == task.ClassBName);

fprintf("\nTask rows: %d\n", numel(yBin));
fprintf("Class A (%s): %d\n", task.ClassAName, sum(yBin == 0));
fprintf("Class B (%s): %d\n", task.ClassBName, sum(yBin == 1));

%% =========================================================
% 3. Select final EEG signature
%% =========================================================

switch selectedFeatureMode
    case "Manual"
        selectedFeatures = manualSelectedFeatures(:);
    case "FromValidation"
        selectedFeatures = loadSelectedFeaturesFromValidation(validationOutDir, signatureName);
    otherwise
        error("Unknown selectedFeatureMode: %s", selectedFeatureMode);
end

selectedFeatures = string(selectedFeatures(:));
selectedIdx = zeros(numel(selectedFeatures), 1);
for i = 1:numel(selectedFeatures)
    idx = find(featureVars == selectedFeatures(i), 1);
    if isempty(idx)
        error("Selected feature not found: %s", selectedFeatures(i));
    end
    selectedIdx(i) = idx;
end

X_task = X_task_allFeatures(:, selectedIdx);
selectedFeatureNames = featureVars(selectedIdx);

fprintf("\nSelected signature features (%d):\n", numel(selectedFeatureNames));
disp(selectedFeatureNames(:));

%% =========================================================
% 4. Train final Bayesian logistic model for explanation
%% =========================================================

[Xz, muX, sigmaX] = zscoreAll(X_task);
sampleWeights = makeClassWeights(yBin);
mdl = trainBayesianLogisticLaplaceWeighted(Xz, yBin, sampleWeights, bayesOpt);

rng(randomSeed);
betaSamples = sampleMVN(mdl.betaMAP(:)', mdl.covBeta, nPosteriorSamples);
betaFeatureSamples = betaSamples(:, 2:end);

predAll = predictWithGivenBetaSamples(betaSamples, Xz);
yhatAll = double(predAll.probMean >= 0.5);

coefSummary = summarizeCoefficientSamples(betaFeatureSamples, selectedFeatureNames);
writetable(coefSummary, fullfile(outDir, "BOPA_CoefficientPosteriorSummary.xlsx"));

fprintf("\nFinal explanation model trained. Iterations: %d\n", mdl.nIter);

%% =========================================================
% 5. Select representative cases
%% =========================================================

if ~isempty(manualTaskRowIndices)
    repIdx = manualTaskRowIndices(:);
    repIdx = repIdx(repIdx >= 1 & repIdx <= size(Xz, 1));
else
    repIdx = selectRepresentativeCases(yBin, yhatAll, predAll, representativeModes);
end
repIdx = unique(repIdx, "stable");

caseTable = buildCaseTable(DATA_task, repIdx, Y, yBin, yhatAll, predAll, task);
writetable(caseTable, fullfile(outDir, "BOPA_RepresentativeCases.xlsx"));

fprintf("\nRepresentative cases:\n");
disp(caseTable(:, ["CaseLabel", "TrueLabel", "PredictedLabel", "ProbMean_Decline", "ProbStd"]));

%% =========================================================
% 6. Baseline attribution distribution
%% =========================================================

allBaselineAttr = table();

for c = 1:numel(repIdx)
    rowIdx = repIdx(c);
    x0 = Xz(rowIdx, :);
    caseLabel = string(caseTable.CaseLabel(c));

    probSamples0 = sigmoidStable([1, x0] * betaSamples');
    probMean0 = mean(probSamples0);
    probStd0 = std(probSamples0);

    % Logit contribution: beta_j * x_j
    logitContributionSamples = betaFeatureSamples .* x0;

    % Probability-level local attribution: x_j * beta_j * p(1-p)
    probSensitivity = probSamples0(:) .* (1 - probSamples0(:));
    probAttributionSamples = logitContributionSamples .* probSensitivity;

    Tlogit = summarizeContributionSamples( ...
        logitContributionSamples, selectedFeatureNames, caseLabel, ...
        "LogitContribution");
    Tprob = summarizeContributionSamples( ...
        probAttributionSamples, selectedFeatureNames, caseLabel, ...
        "ProbabilityAttribution");

    Tlogit.ProbMean_Decline(:) = probMean0;
    Tlogit.ProbStd(:) = probStd0;
    Tprob.ProbMean_Decline(:) = probMean0;
    Tprob.ProbStd(:) = probStd0;

    allBaselineAttr = [allBaselineAttr; Tlogit; Tprob]; %#ok<AGROW>

    plotAttributionBarWithUncertainty(Tlogit, figDir, caseLabel, "LogitContribution");
    plotAttributionBarWithUncertainty(Tprob, figDir, caseLabel, "ProbabilityAttribution");
end

writetable(allBaselineAttr, fullfile(outDir, "BOPA_BaselineAttributionDistribution.xlsx"));

%% =========================================================
% 7. One-feature output-preserving sweep
%% =========================================================

allSweepRows = table();
allRangeRows = table();

for c = 1:numel(repIdx)
    rowIdx = repIdx(c);
    x0 = Xz(rowIdx, :);
    caseLabel = string(caseTable.CaseLabel(c));

    baselineProbSamples = sigmoidStable([1, x0] * betaSamples');
    baselineProbMean = mean(baselineProbSamples);
    baselineClass = double(baselineProbMean >= 0.5);

    fprintf("\nSweeping features for %s | baseline P(Decline)=%.3f | class=%d\n", ...
        caseLabel, baselineProbMean, baselineClass);

    [sweepRows, rangeRows] = runOneFeatureSweep( ...
        x0, betaSamples, betaFeatureSamples, selectedFeatureNames, ...
        muX, sigmaX, Xz, sweepOffsetZ, ...
        baselineProbMean, baselineClass, ...
        minPosteriorAgreement, useConfidencePreserving, maxProbMeanChange, ...
        clipSweepToObservedRange, observedPrctileRange, caseLabel);

    allSweepRows = [allSweepRows; sweepRows]; %#ok<AGROW>
    allRangeRows = [allRangeRows; rangeRows]; %#ok<AGROW>

    plotOutputPreservingRangeBar(rangeRows, figDir, caseLabel);
    plotContributionSurface(sweepRows, selectedFeatureNames, figDir, caseLabel);
    plotPredictiveUncertaintySurface(sweepRows, selectedFeatureNames, figDir, caseLabel);
    plotOutputPreservingMask(sweepRows, selectedFeatureNames, figDir, caseLabel);
    plotProbabilityCurvesByFeature(sweepRows, selectedFeatureNames, figDir, caseLabel);
end

writetable(allSweepRows, fullfile(outDir, "BOPA_OutputPreservingSweep_AllRows.xlsx"));
writetable(allRangeRows, fullfile(outDir, "BOPA_OutputPreservingRange_ByFeature.xlsx"));

rangeSummary = summarizeRangeAcrossCases(allRangeRows);
writetable(rangeSummary, fullfile(outDir, "BOPA_OutputPreservingRange_GroupSummary.xlsx"));
plotRangeGroupSummary(rangeSummary, figDir);

%% =========================================================
% 8. Save model package
%% =========================================================

modelPackage = struct();
modelPackage.task = task;
modelPackage.selectedFeatureMode = selectedFeatureMode;
modelPackage.signatureName = signatureName;
modelPackage.selectedFeatureNames = selectedFeatureNames;
modelPackage.muX = muX;
modelPackage.sigmaX = sigmaX;
modelPackage.bayesOpt = bayesOpt;
modelPackage.mdl = mdl;
modelPackage.betaSamples = betaSamples;
modelPackage.caseTable = caseTable;
modelPackage.note = "Trained on all task rows for explanation visualization, not unbiased performance estimation.";

save(fullfile(outDir, "BOPA_OutputPreservingAttribution_ModelPackage.mat"), ...
    "modelPackage", "-v7.3");

fprintf("\nDone.\n");
fprintf("Tables saved to: %s\n", outDir);
fprintf("Figures saved to: %s\n", figDir);

%% =========================================================
% Local functions: loading
%% =========================================================

function selectedFeatures = loadSelectedFeaturesFromValidation(validationOutDir, signatureName)
    rawFile = fullfile(validationOutDir, "FixedSignature_RawMetrics_RepeatedCV.xlsx");
    if exist(rawFile, "file") == 2
        T = readtable(rawFile, "VariableNamingRule", "preserve");
        T.SignatureName = string(T.SignatureName);
        rows = T(T.SignatureName == signatureName, :);
        if ~isempty(rows) && ismember("SelectedFeatures", string(T.Properties.VariableNames))
            selectedFeatures = splitCommaString(rows.SelectedFeatures(1));
            return;
        end
    end
    error("Could not load selected features for signature: %s", signatureName);
end

function items = splitCommaString(x)
    x = string(x);
    items = split(x, ",");
    items = strtrim(items);
    items = items(items ~= "" & ~ismissing(items));
end

%% =========================================================
% Local functions: training / prediction
%% =========================================================

function [Xz, muX, sigmaX] = zscoreAll(X)
    muX = mean(X, 1, "omitnan");
    sigmaX = std(X, 0, 1, "omitnan");
    sigmaX(sigmaX == 0 | isnan(sigmaX)) = 1;
    Xz = (X - muX) ./ sigmaX;
end

function X = replaceInvalidWithMedian(X)
    for j = 1:size(X, 2)
        col = X(:, j);
        medVal = median(col, "omitnan");
        if isnan(medVal); medVal = 0; end
        col(~isfinite(col)) = medVal;
        X(:, j) = col;
    end
end

function w = makeClassWeights(y)
    y = double(y(:));
    n0 = sum(y == 0);
    n1 = sum(y == 1);
    n = numel(y);
    w0 = n / max(2 * n0, 1);
    w1 = n / max(2 * n1, 1);
    w = zeros(size(y));
    w(y == 0) = w0;
    w(y == 1) = w1;
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

    prob = sigmoidStable(Xb * beta);
    W = sampleWeights .* prob .* (1 - prob);
    W = max(W, 1e-8);
    H = Xb' * (Xb .* W) + diag(priorPrec);
    H = H + opt.jitter * eye(p + 1);

    mdl = struct();
    mdl.betaMAP = beta;
    mdl.covBeta = safeInverseSPD(H, opt.jitter);
    mdl.priorStd = opt.priorStd;
    mdl.interceptPriorStd = opt.interceptPriorStd;
    mdl.nIter = iter;
end

function pred = predictWithGivenBetaSamples(betaSamples, X)
    Xb = [ones(size(X, 1), 1), X];
    probSamples = sigmoidStable(Xb * betaSamples');
    pred = struct();
    pred.probSamples = probSamples;
    pred.probMean = mean(probSamples, 2);
    pred.probStd = std(probSamples, 0, 2);
    pred.probCI = prctile(probSamples, [2.5, 97.5], 2);
end

function X = sampleMVN(mu, C, n)
    mu = double(mu(:)');
    C = double(C);
    C = (C + C') / 2;
    p = numel(mu);
    [R, flag] = chol(C, "lower");
    if flag ~= 0
        jitter = 1e-8;
        for k = 1:8
            [R, flag] = chol(C + jitter * eye(p), "lower");
            if flag == 0; break; end
            jitter = jitter * 10;
        end
    end
    if flag ~= 0
        warning("Cholesky failed. Using diagonal covariance approximation.");
        d = max(diag(C), 1e-8);
        R = diag(sqrt(d));
    end
    X = randn(n, p) * R' + mu;
end

function invA = safeInverseSPD(A, jitter)
    A = (A + A') / 2;
    p = size(A, 1);
    [R, flag] = chol(A, "lower");
    if flag ~= 0
        jj = jitter;
        for k = 1:8
            [R, flag] = chol(A + jj * eye(p), "lower");
            if flag == 0; break; end
            jj = jj * 10;
        end
    end
    if flag == 0
        invR = R \ eye(p);
        invA = invR' * invR;
    else
        warning("safeInverseSPD: Cholesky failed. Using pinv.");
        invA = pinv(A);
    end
    invA = (invA + invA') / 2;
end

function p = sigmoidStable(z)
    p = zeros(size(z));
    idx = z >= 0;
    p(idx) = 1 ./ (1 + exp(-z(idx)));
    ez = exp(z(~idx));
    p(~idx) = ez ./ (1 + ez);
    p = min(max(p, 1e-12), 1 - 1e-12);
end

function h = binaryEntropy(p)
    p = min(max(p, 1e-12), 1 - 1e-12);
    h = -(p .* log(p) + (1 - p) .* log(1 - p));
end

%% =========================================================
% Local functions: representative cases
%% =========================================================

function repIdx = selectRepresentativeCases(y, yhat, pred, representativeModes)
    repIdx = [];
    for i = 1:numel(representativeModes)
        modeName = representativeModes(i);
        switch modeName
            case "ConfidentMaintain"
                cand = find(y == 0 & yhat == 0);
                if isempty(cand); cand = find(y == 0); end
                [~, k] = min(pred.probMean(cand));
                repIdx(end+1, 1) = cand(k); %#ok<AGROW>
            case "ConfidentDecline"
                cand = find(y == 1 & yhat == 1);
                if isempty(cand); cand = find(y == 1); end
                [~, k] = max(pred.probMean(cand));
                repIdx(end+1, 1) = cand(k); %#ok<AGROW>
            case "Boundary"
                [~, k] = min(abs(pred.probMean - 0.5));
                repIdx(end+1, 1) = k; %#ok<AGROW>
            case "HighUncertainty"
                [~, k] = max(pred.probStd);
                repIdx(end+1, 1) = k; %#ok<AGROW>
        end
    end
end

function caseTable = buildCaseTable(DATA_task, repIdx, Y, yBin, yhat, pred, task)
    n = numel(repIdx);
    caseTable = table();
    caseTable.CaseLabel = strings(n, 1);
    caseTable.TaskRowIndex = repIdx(:);
    if ismember("ObjectID", string(DATA_task.Properties.VariableNames))
        caseTable.ObjectID = string(DATA_task.ObjectID(repIdx));
    else
        caseTable.ObjectID = repmat("", n, 1);
    end
    if ismember("FileKey", string(DATA_task.Properties.VariableNames))
        caseTable.FileKey = string(DATA_task.FileKey(repIdx));
    else
        caseTable.FileKey = repmat("", n, 1);
    end
    caseTable.TrueLabel = Y(repIdx);
    caseTable.TrueBinary = yBin(repIdx);
    caseTable.PredictedBinary = yhat(repIdx);
    caseTable.PredictedLabel = strings(n, 1);
    caseTable.PredictedLabel(caseTable.PredictedBinary == 0) = task.ClassAName;
    caseTable.PredictedLabel(caseTable.PredictedBinary == 1) = task.ClassBName;
    caseTable.ProbMean_Decline = pred.probMean(repIdx);
    caseTable.ProbStd = pred.probStd(repIdx);
    caseTable.ProbCI025 = pred.probCI(repIdx, 1);
    caseTable.ProbCI975 = pred.probCI(repIdx, 2);
    caseTable.Entropy = binaryEntropy(caseTable.ProbMean_Decline);
    for i = 1:n
        caseTable.CaseLabel(i) = "Case" + i + "_row" + repIdx(i);
    end
end

%% =========================================================
% Local functions: summaries
%% =========================================================

function summaryT = summarizeCoefficientSamples(betaFeatureSamples, featureNames)
    ci = prctile(betaFeatureSamples, [2.5, 97.5], 1);
    summaryT = table();
    summaryT.Feature = string(featureNames(:));
    summaryT.BetaMean = mean(betaFeatureSamples, 1)';
    summaryT.BetaStd = std(betaFeatureSamples, 0, 1)';
    summaryT.BetaCI025 = ci(1, :)';
    summaryT.BetaCI975 = ci(2, :)';
    summaryT.ProbPositiveEffect = mean(betaFeatureSamples > 0, 1)';
    summaryT.AbsZ = abs(summaryT.BetaMean) ./ max(summaryT.BetaStd, eps);
    direction = strings(height(summaryT), 1);
    direction(summaryT.ProbPositiveEffect >= 0.975) = "Stable_Decline_direction";
    direction(summaryT.ProbPositiveEffect <= 0.025) = "Stable_Maintain_direction";
    direction(direction == "") = "Uncertain_or_mixed";
    summaryT.Direction = direction;
    summaryT = sortrows(summaryT, "AbsZ", "descend");
end

function T = summarizeContributionSamples(contributionSamples, featureNames, caseLabel, contributionType)
    ci = prctile(contributionSamples, [2.5, 97.5], 1);
    T = table();
    T.CaseLabel = repmat(string(caseLabel), numel(featureNames), 1);
    T.ContributionType = repmat(string(contributionType), numel(featureNames), 1);
    T.Feature = string(featureNames(:));
    T.MeanContribution = mean(contributionSamples, 1)';
    T.StdContribution = std(contributionSamples, 0, 1)';
    T.CI025 = ci(1, :)';
    T.CI975 = ci(2, :)';
    T.ProbPositiveContribution = mean(contributionSamples > 0, 1)';
    T.AbsMeanOverStd = abs(T.MeanContribution) ./ max(T.StdContribution, eps);
    T = sortrows(T, "AbsMeanOverStd", "descend");
end

%% =========================================================
% Local functions: sweep
%% =========================================================

function [sweepRows, rangeRows] = runOneFeatureSweep( ...
    x0, betaSamples, betaFeatureSamples, featureNames, ...
    muX, sigmaX, XzAll, sweepOffsetZ, ...
    baselineProbMean, baselineClass, ...
    minPosteriorAgreement, useConfidencePreserving, maxProbMeanChange, ...
    clipSweepToObservedRange, observedPrctileRange, caseLabel)

    nFeat = numel(featureNames);
    nGrid = numel(sweepOffsetZ);
    sweepRows = table();
    rangeRows = table();

    for j = 1:nFeat
        featureName = string(featureNames(j));
        baseZ = x0(j);
        zValues = baseZ + sweepOffsetZ(:);

        if clipSweepToObservedRange
            loObs = prctile(XzAll(:, j), observedPrctileRange(1));
            hiObs = prctile(XzAll(:, j), observedPrctileRange(2));
            validZ = zValues >= loObs & zValues <= hiObs;
        else
            validZ = true(size(zValues));
        end

        rowsThisFeature = table();

        for k = 1:nGrid
            zVal = zValues(k);
            row = table();
            row.CaseLabel = string(caseLabel);
            row.FeatureIndex = j;
            row.Feature = featureName;
            row.SweepIndex = k;
            row.SweepOffsetZ = sweepOffsetZ(k);
            row.SweepValueZ = zVal;
            row.SweepValueOriginal = muX(j) + zVal * sigmaX(j);
            row.IsWithinObservedRange = validZ(k);

            if ~validZ(k)
                row.ProbMean_Decline = NaN;
                row.ProbStd = NaN;
                row.ProbCI025 = NaN;
                row.ProbCI975 = NaN;
                row.PosteriorAgreement = NaN;
                row.LogitContributionMean = NaN;
                row.LogitContributionStd = NaN;
                row.LogitContributionCI025 = NaN;
                row.LogitContributionCI975 = NaN;
                row.ProbabilityAttributionMean = NaN;
                row.ProbabilityAttributionStd = NaN;
                row.ProbabilityAttributionCI025 = NaN;
                row.ProbabilityAttributionCI975 = NaN;
                row.IsClassPreserving = false;
                row.IsPosteriorAgreementPreserving = false;
                row.IsConfidencePreserving = false;
                row.IsOutputPreserving = false;
                rowsThisFeature = [rowsThisFeature; row]; %#ok<AGROW>
                continue;
            end

            xNew = x0;
            xNew(j) = zVal;
            probSamples = sigmoidStable([1, xNew] * betaSamples');
            probMean = mean(probSamples);
            probStd = std(probSamples);
            probCI = prctile(probSamples, [2.5, 97.5]);

            if baselineClass == 1
                classPreserving = probMean >= 0.5;
                posteriorAgreement = mean(probSamples >= 0.5);
            else
                classPreserving = probMean < 0.5;
                posteriorAgreement = mean(probSamples < 0.5);
            end

            agreementPreserving = posteriorAgreement >= minPosteriorAgreement;
            if useConfidencePreserving
                confidencePreserving = abs(probMean - baselineProbMean) <= maxProbMeanChange;
            else
                confidencePreserving = true;
            end
            outputPreserving = classPreserving && agreementPreserving && confidencePreserving;

            logitSamples = betaFeatureSamples(:, j) .* zVal;
            probSensitivity = probSamples(:) .* (1 - probSamples(:));
            probAttrSamples = logitSamples .* probSensitivity;
            logitCI = prctile(logitSamples, [2.5, 97.5]);
            probAttrCI = prctile(probAttrSamples, [2.5, 97.5]);

            row.ProbMean_Decline = probMean;
            row.ProbStd = probStd;
            row.ProbCI025 = probCI(1);
            row.ProbCI975 = probCI(2);
            row.PosteriorAgreement = posteriorAgreement;
            row.LogitContributionMean = mean(logitSamples);
            row.LogitContributionStd = std(logitSamples);
            row.LogitContributionCI025 = logitCI(1);
            row.LogitContributionCI975 = logitCI(2);
            row.ProbabilityAttributionMean = mean(probAttrSamples);
            row.ProbabilityAttributionStd = std(probAttrSamples);
            row.ProbabilityAttributionCI025 = probAttrCI(1);
            row.ProbabilityAttributionCI975 = probAttrCI(2);
            row.IsClassPreserving = classPreserving;
            row.IsPosteriorAgreementPreserving = agreementPreserving;
            row.IsConfidencePreserving = confidencePreserving;
            row.IsOutputPreserving = outputPreserving;

            rowsThisFeature = [rowsThisFeature; row]; %#ok<AGROW>
        end

        sweepRows = [sweepRows; rowsThisFeature]; %#ok<AGROW>

        validPreserve = rowsThisFeature.IsOutputPreserving & rowsThisFeature.IsWithinObservedRange;
        rangeRow = table();
        rangeRow.CaseLabel = string(caseLabel);
        rangeRow.FeatureIndex = j;
        rangeRow.Feature = featureName;
        rangeRow.BaselineValueZ = baseZ;
        rangeRow.BaselineValueOriginal = muX(j) + baseZ * sigmaX(j);

        if any(validPreserve)
            zPreserve = rowsThisFeature.SweepValueZ(validPreserve);
            origPreserve = rowsThisFeature.SweepValueOriginal(validPreserve);
            rangeRow.PreserveMinZ = min(zPreserve);
            rangeRow.PreserveMaxZ = max(zPreserve);
            rangeRow.PreserveRangeZ = max(zPreserve) - min(zPreserve);
            rangeRow.PreserveMinOriginal = min(origPreserve);
            rangeRow.PreserveMaxOriginal = max(origPreserve);
            rangeRow.PreserveRangeOriginal = max(origPreserve) - min(origPreserve);
            rangeRow.PreserveGridCount = sum(validPreserve);
        else
            rangeRow.PreserveMinZ = NaN;
            rangeRow.PreserveMaxZ = NaN;
            rangeRow.PreserveRangeZ = 0;
            rangeRow.PreserveMinOriginal = NaN;
            rangeRow.PreserveMaxOriginal = NaN;
            rangeRow.PreserveRangeOriginal = 0;
            rangeRow.PreserveGridCount = 0;
        end

        baseLogitSamples = betaFeatureSamples(:, j) .* baseZ;
        baseProbSamples = sigmoidStable([1, x0] * betaSamples');
        baseProbSensitivity = baseProbSamples(:) .* (1 - baseProbSamples(:));
        baseProbAttrSamples = baseLogitSamples .* baseProbSensitivity;

        rangeRow.BaselineLogitContributionMean = mean(baseLogitSamples);
        rangeRow.BaselineLogitContributionStd = std(baseLogitSamples);
        rangeRow.BaselineProbabilityAttributionMean = mean(baseProbAttrSamples);
        rangeRow.BaselineProbabilityAttributionStd = std(baseProbAttrSamples);

        rangeRows = [rangeRows; rangeRow]; %#ok<AGROW>
    end
end

function rangeSummary = summarizeRangeAcrossCases(allRangeRows)
    [G, groupT] = findgroups(allRangeRows(:, ["FeatureIndex", "Feature"]));
    rangeSummary = groupT;
    rangeSummary.MeanPreserveRangeZ = splitapply(@(x) mean(x, "omitnan"), allRangeRows.PreserveRangeZ, G);
    rangeSummary.StdPreserveRangeZ = splitapply(@(x) std(x, 0, "omitnan"), allRangeRows.PreserveRangeZ, G);
    rangeSummary.MedianPreserveRangeZ = splitapply(@(x) median(x, "omitnan"), allRangeRows.PreserveRangeZ, G);
    rangeSummary.MeanPreserveRangeOriginal = splitapply(@(x) mean(x, "omitnan"), allRangeRows.PreserveRangeOriginal, G);
    rangeSummary.MeanBaselineLogitContributionAbs = splitapply(@(x) mean(abs(x), "omitnan"), allRangeRows.BaselineLogitContributionMean, G);
    rangeSummary.MeanBaselineLogitUncertainty = splitapply(@(x) mean(x, "omitnan"), allRangeRows.BaselineLogitContributionStd, G);
    rangeSummary.MeanBaselineProbAttrAbs = splitapply(@(x) mean(abs(x), "omitnan"), allRangeRows.BaselineProbabilityAttributionMean, G);
    rangeSummary.MeanBaselineProbAttrUncertainty = splitapply(@(x) mean(x, "omitnan"), allRangeRows.BaselineProbabilityAttributionStd, G);
    rangeSummary = sortrows(rangeSummary, "MeanBaselineLogitContributionAbs", "descend");
end

%% =========================================================
% Local functions: plots
%% =========================================================

function plotAttributionBarWithUncertainty(T, figDir, caseLabel, contributionType)
    T = sortrows(T, "MeanContribution", "ascend");
    y = T.MeanContribution;
    lo = T.CI025;
    hi = T.CI975;
    fig = figure("Color", "w", "Position", [100, 100, 1050, 560]);
    barh(y);
    hold on;
    for i = 1:numel(y)
        plot([lo(i), hi(i)], [i, i], "k-", "LineWidth", 1.2);
    end
    xline(0, "--", "LineWidth", 1.0);
    hold off;
    grid on; box off;
    yticks(1:height(T));
    yticklabels(shortenFeatureNames(T.Feature));
    xlabel(contributionType);
    title("Bayesian attribution distribution: " + caseLabel + " / " + contributionType, "Interpreter", "none");
    saveFigure(fig, figDir, "Fig01_BaselineAttribution_" + safeName(contributionType) + "_" + safeName(caseLabel));
end

function plotOutputPreservingRangeBar(rangeRows, figDir, caseLabel)
    T = sortrows(rangeRows, "FeatureIndex", "ascend");
    fig = figure("Color", "w", "Position", [100, 100, 1050, 560]);
    yyaxis left;
    bar(T.PreserveRangeZ);
    ylabel("Output-preserving range length (z-score units)");
    yyaxis right;
    plot(abs(T.BaselineLogitContributionMean), "o-", "LineWidth", 1.4);
    hold on;
    plot(T.BaselineLogitContributionStd, "s--", "LineWidth", 1.2);
    hold off;
    ylabel("Baseline contribution / uncertainty");
    grid on; box off;
    xticks(1:height(T));
    xticklabels(shortenFeatureNames(T.Feature));
    xtickangle(35);
    title("Output-preserving feature range: " + caseLabel, "Interpreter", "none");
    legend(["Preserving range", "|baseline contribution|", "contribution uncertainty"], ...
        "Location", "southoutside", "Orientation", "horizontal");
    saveFigure(fig, figDir, "Fig02_OutputPreservingRange_" + safeName(caseLabel));
end

function plotContributionSurface(sweepRows, featureNames, figDir, caseLabel)
    [Xgrid, Ygrid, Zgrid, Cgrid] = makeSurfaceMatrices(sweepRows, featureNames, "LogitContributionMean", "LogitContributionStd");
    fig = figure("Color", "w", "Position", [100, 100, 1100, 650]);
    surf(Xgrid, Ygrid, Zgrid, Cgrid, "EdgeColor", "none");
    colorbar;
    xlabel("Feature perturbation offset (z-score)");
    ylabel("EEG feature");
    zlabel("Mean logit contribution");
    yticks(1:numel(featureNames));
    yticklabels(shortenFeatureNames(featureNames));
    title("Contribution surface with uncertainty color: " + caseLabel, "Interpreter", "none");
    grid on; view(45, 28);
    saveFigure(fig, figDir, "Fig03_ContributionSurface_" + safeName(caseLabel));
end

function plotPredictiveUncertaintySurface(sweepRows, featureNames, figDir, caseLabel)
    [Xgrid, Ygrid, Zgrid, Cgrid] = makeSurfaceMatrices(sweepRows, featureNames, "ProbStd", "ProbMean_Decline");
    fig = figure("Color", "w", "Position", [100, 100, 1100, 650]);
    surf(Xgrid, Ygrid, Zgrid, Cgrid, "EdgeColor", "none");
    colorbar;
    xlabel("Feature perturbation offset (z-score)");
    ylabel("EEG feature");
    zlabel("Predictive uncertainty: std[P(Decline)]");
    yticks(1:numel(featureNames));
    yticklabels(shortenFeatureNames(featureNames));
    title("Predictive uncertainty surface with probability color: " + caseLabel, "Interpreter", "none");
    grid on; view(45, 28);
    saveFigure(fig, figDir, "Fig04_PredictiveUncertaintySurface_" + safeName(caseLabel));
end

function plotOutputPreservingMask(sweepRows, featureNames, figDir, caseLabel)
    [Xgrid, ~, Zgrid, ~] = makeSurfaceMatrices(sweepRows, featureNames, "IsOutputPreserving", "ProbMean_Decline");
    fig = figure("Color", "w", "Position", [100, 100, 1050, 560]);
    imagesc(Xgrid(1, :), 1:numel(featureNames), Zgrid);
    colorbar;
    xlabel("Feature perturbation offset (z-score)");
    ylabel("EEG feature");
    yticks(1:numel(featureNames));
    yticklabels(shortenFeatureNames(featureNames));
    title("Output-preserving region mask: " + caseLabel, "Interpreter", "none");
    set(gca, "YDir", "normal");
    saveFigure(fig, figDir, "Fig05_OutputPreservingMask_" + safeName(caseLabel));
end

function plotProbabilityCurvesByFeature(sweepRows, featureNames, figDir, caseLabel)
    nFeat = numel(featureNames);
    nCol = min(3, nFeat);
    nRow = ceil(nFeat / nCol);
    fig = figure("Color", "w", "Position", [100, 100, 1150, max(500, 250*nRow)]);
    tiledlayout(nRow, nCol, "TileSpacing", "compact", "Padding", "compact");
    for j = 1:nFeat
        T = sweepRows(sweepRows.FeatureIndex == j, :);
        T = sortrows(T, "SweepOffsetZ");
        nexttile;
        plot(T.SweepOffsetZ, T.ProbMean_Decline, "-", "LineWidth", 1.5);
        hold on;
        plot(T.SweepOffsetZ, T.ProbCI025, "--", "LineWidth", 1.0);
        plot(T.SweepOffsetZ, T.ProbCI975, "--", "LineWidth", 1.0);
        preserveIdx = T.IsOutputPreserving;
        if any(preserveIdx)
            scatter(T.SweepOffsetZ(preserveIdx), T.ProbMean_Decline(preserveIdx), 20, "filled");
        end
        yline(0.5, ":", "LineWidth", 1.0);
        xline(0, ":", "LineWidth", 1.0);
        hold off;
        ylim([0, 1]);
        grid on; box off;
        title(shortenFeatureNames(featureNames(j)), "Interpreter", "none");
        xlabel("offset z");
        ylabel("P(Decline)");
    end
    sgtitle("One-feature probability curves: " + caseLabel, "Interpreter", "none");
    saveFigure(fig, figDir, "Fig06_ProbabilityCurves_" + safeName(caseLabel));
end

function plotRangeGroupSummary(rangeSummary, figDir)
    T = sortrows(rangeSummary, "MeanBaselineLogitContributionAbs", "descend");
    fig = figure("Color", "w", "Position", [100, 100, 1050, 560]);
    yyaxis left;
    bar(T.MeanPreserveRangeZ);
    ylabel("Mean output-preserving range length (z-score)");
    yyaxis right;
    plot(T.MeanBaselineLogitContributionAbs, "o-", "LineWidth", 1.4);
    hold on;
    plot(T.MeanBaselineLogitUncertainty, "s--", "LineWidth", 1.2);
    hold off;
    ylabel("Mean baseline contribution / uncertainty");
    grid on; box off;
    xticks(1:height(T));
    xticklabels(shortenFeatureNames(T.Feature));
    xtickangle(35);
    title("Group summary across representative cases");
    legend(["Mean preserving range", "Mean |contribution|", "Mean contribution uncertainty"], ...
        "Location", "southoutside", "Orientation", "horizontal");
    saveFigure(fig, figDir, "Fig07_GroupRangeSummary");
end

function [Xgrid, Ygrid, Zgrid, Cgrid] = makeSurfaceMatrices(sweepRows, featureNames, zVar, cVar)
    offsets = unique(sweepRows.SweepOffsetZ, "stable");
    nFeat = numel(featureNames);
    nGrid = numel(offsets);
    Xgrid = repmat(offsets(:)', nFeat, 1);
    Ygrid = repmat((1:nFeat)', 1, nGrid);
    Zgrid = nan(nFeat, nGrid);
    Cgrid = nan(nFeat, nGrid);
    for j = 1:nFeat
        T = sweepRows(sweepRows.FeatureIndex == j, :);
        for k = 1:nGrid
            row = T(T.SweepOffsetZ == offsets(k), :);
            if ~isempty(row)
                Zgrid(j, k) = double(row.(zVar)(1));
                Cgrid(j, k) = double(row.(cVar)(1));
            end
        end
    end
end

function saveFigure(fig, figDir, baseName)
    pngPath = fullfile(figDir, baseName + ".png");
    pdfPath = fullfile(figDir, baseName + ".pdf");
    try
        exportgraphics(fig, pngPath, "Resolution", 300);
        exportgraphics(fig, pdfPath, "ContentType", "vector");
    catch
        saveas(fig, pngPath);
        saveas(fig, pdfPath);
    end
    fprintf("Saved: %s\n", pngPath);
end

function labels = shortenFeatureNames(features)
    labels = string(features);
    labels = strrep(labels, "theta_alpha_ratio", "theta/alpha");
    labels = strrep(labels, "median_freq", "med.freq");
    labels = strrep(labels, "theta_rel", "theta");
    labels = strrep(labels, "alpha_rel", "alpha");
    labels = strrep(labels, "beta_rel", "beta");
    labels = strrep(labels, "ch1_ch2_corr", "corr");
    labels = strrep(labels, "mean_", "avg ");
    labels = strrep(labels, "sef95", "SEF95");
    labels = strrep(labels, "_", " ");
end

function nameOut = safeName(x)
    nameOut = string(x);
    nameOut = regexprep(nameOut, "[^\w\d가-힣_-]", "_");
end

%% =========================================================
% Local functions: diagnosis normalization
%% =========================================================

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
            if contains(sub, "CN_to_MCI") || contains(sub, "SCD_to_MCI")
                subtype(i) = "Decline_to_MCI";
            elseif contains(sub, "CN_to_DEM") || contains(sub, "SCD_to_DEM")
                subtype(i) = "Decline_Normal_to_DEM";
            elseif contains(sub, "MCI_to_DEM")
                subtype(i) = "Decline_MCI_to_DEM";
            else
                subtype(i) = "Decline_Other";
            end
        elseif lab == "Impaired_Stable"
            subtype(i) = "Impaired_Stable";
        elseif lab == "Improving_or_Mixed"
            subtype(i) = "Improving_or_Mixed";
        else
            subtype(i) = "Other";
        end
    end
end
