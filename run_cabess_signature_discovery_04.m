%% 04_run_cabess_signature_discovery.m
% CaBESS: class-aware Bayesian evolutionary EEG signature discovery.
% Class-aware Bayesian guided evolutionary EEG signature selection
%
% Purpose:
% 1) Build Decline-specialist parent models and Maintain-specialist parent models
% 2) Cross class-specialist parents to generate balanced child signatures
% 3) Use Bayesian uncertainty-guided mutation rather than purely random mutation
% 4) Identify compact EEG feature signatures for dementia progression prediction
%
% Task:
% Maintain_CN_SCD vs Decline_to_DEM
%
% Methodological note:
% This script performs signature discovery. Search-stage CV values are search
% diagnostics and should not be interpreted as a fully nested estimate of
% generalization performance. The selected fixed signature is evaluated
% separately by 05_validate_fixed_cabess_signature.m.

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

outDir = fullfile(paths.outputRoot, "04_cabess_discovery");

if ~exist(outDir, "dir")
    mkdir(outDir);
end

rng(42);

fprintf("\n=== CaBESS signature discovery ===\n");
fprintf("Feature file: %s\n", featureFile);
fprintf("Output dir  : %s\n", outDir);

%% =========================================================
% 1. Task and experiment setting
%% =========================================================

task.Name = "Subtype_MaintainNormal_vs_DeclineToDEM";
task.SourceVar = "AnalysisSubtype";
task.ClassAValues = ["Maintain_CN", "Maintain_SCD"];
task.ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM"];
task.ClassAName = "Maintain_CN_SCD";
task.ClassBName = "Decline_to_DEM";

positiveLabel = string(task.ClassBName);

% Main validation setting
% Candidate evaluation uses subject-level five-fold CV during signature discovery.
% Final fixed-signature repeated validation is performed by script 05.
splitMode = "SubjectLevel";
kfoldDefault = 5;

% Bayesian logistic setting
bayesOpt = struct();
bayesOpt.priorStd = 0.5;
bayesOpt.interceptPriorStd = 20;
bayesOpt.maxIter = 200;
bayesOpt.tol = 1e-6;
bayesOpt.jitter = 1e-6;

% Posterior sampling
nPosteriorSamplesEval = 300;
nPosteriorSamplesFinal = 1000;

% Evolutionary search setting
opt = struct();
opt.populationSize = 24;
opt.nSpecialistGenerations = 12;
opt.nChildGenerations = 15;
opt.eliteCount = 3;
opt.tournamentSize = 3;
opt.minFeatures = 4;
opt.maxFeatures = 14;

% Mutation setting
opt.guidedMutationRate = 0.70;      % probability of guided mutation
opt.randomFlipRate = 0.015;         % small random exploration
opt.addProb = 0.55;
opt.dropProb = 0.25;
opt.swapProb = 0.20;

% Specialist parent should not become a trivial one-class predictor.
opt.minOppositeRecall = 0.45;

% Guide setting
opt.topGuideFraction = 0.35;        % candidate pool size from global separation score
opt.eliteGuideBoost = 1.00;         % boost features selected in specialist elites
opt.groupDiversityWeight = 0.35;    % favor underrepresented feature groups during add mutation

% Fitness penalties
opt.probStdScale = 0.15;
opt.attrUncScale = 0.05;
opt.redundancyScale = 0.60;

%% =========================================================
% 2. Load data
%% =========================================================

DATA = readtable(featureFile, "VariableNamingRule", "preserve");

requiredVars = ["Strict_Label", "Strict_Subtype", "Current_Diagnosis"];
for i = 1:numel(requiredVars)
    if ~ismember(requiredVars(i), string(DATA.Properties.VariableNames))
        error("%s column is required.", requiredVars(i));
    end
end

DATA.Strict_Label = string(strtrim(DATA.Strict_Label));
DATA.Strict_Subtype = string(strtrim(DATA.Strict_Subtype));
DATA.Current_Diagnosis = normalizeDx(DATA.Current_Diagnosis);

DATA.AnalysisSubtype = simplifyProgressionSubtype( ...
    DATA.Strict_Label, DATA.Strict_Subtype, DATA.Current_Diagnosis);

fprintf("\nRows: %d\n", height(DATA));

%% =========================================================
% 3. Build feature matrix
%% =========================================================

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
X = table2array(DATA(:, cellstr(featureVars)));

for j = 1:size(X, 2)
    col = X(:, j);
    medVal = median(col, "omitnan");
    if isnan(medVal)
        medVal = 0;
    end
    col(~isfinite(col)) = medVal;
    X(:, j) = col;
end

fprintf("Number of numeric features: %d\n", numel(featureVars));

%% =========================================================
% 4. Select target task
%% =========================================================

source = string(DATA.(task.SourceVar));

idxA = ismember(source, task.ClassAValues);
idxB = ismember(source, task.ClassBValues);
keep = idxA | idxB;

DATA_task = DATA(keep, :);
X_task = X(keep, :);

Y = strings(sum(keep), 1);
Y(idxA(keep)) = task.ClassAName;
Y(idxB(keep)) = task.ClassBName;

DATA_task.BinaryLabel = Y;
yBin = double(Y == positiveLabel);

fprintf("\nTask: %s\n", task.Name);
fprintf("Class A [%s] N = %d\n", task.ClassAName, sum(yBin == 0));
fprintf("Class B [%s] N = %d\n", task.ClassBName, sum(yBin == 1));

writetable(countByString(Y, "BinaryLabel"), fullfile(outDir, "ClassCount.xlsx"));

featureGroups = strings(numel(featureVars), 1);
for j = 1:numel(featureVars)
    featureGroups(j) = inferFeatureGroup(string(featureVars(j)));
end

%% =========================================================
% 5. Global guide for Bayesian-guided mutation
%% =========================================================

fprintf("\nComputing global mutation guide...\n");

guide = computeGlobalMutationGuide(X_task, yBin, featureVars, featureGroups, opt);
writetable(guide.table, fullfile(outDir, "Global_MutationGuide.xlsx"));

p = size(X_task, 2);

%% =========================================================
% 6. Specialist parent evolution
%% =========================================================

fprintf("\n=== Stage 1: Specialist parent evolution ===\n");

% Decline-specialist population
popDecline = initializePopulation(p, opt, guide, "Decline");
[popDecline, declineLog, declineResults] = evolveSpecialistPopulation( ...
    popDecline, "DeclineParent", X_task, yBin, DATA_task, featureVars, featureGroups, ...
    guide, splitMode, kfoldDefault, bayesOpt, nPosteriorSamplesEval, opt);

% Maintain-specialist population
popMaintain = initializePopulation(p, opt, guide, "Maintain");
[popMaintain, maintainLog, maintainResults] = evolveSpecialistPopulation( ...
    popMaintain, "MaintainParent", X_task, yBin, DATA_task, featureVars, featureGroups, ...
    guide, splitMode, kfoldDefault, bayesOpt, nPosteriorSamplesEval, opt);

writetable(declineLog, fullfile(outDir, "Specialist_Decline_GenerationLog.xlsx"));
writetable(maintainLog, fullfile(outDir, "Specialist_Maintain_GenerationLog.xlsx"));
writetable(declineResults, fullfile(outDir, "Specialist_Decline_IndividualResults.xlsx"));
writetable(maintainResults, fullfile(outDir, "Specialist_Maintain_IndividualResults.xlsx"));

% Update guide by elite parent feature frequencies.
[declineEliteMasks, declineEliteScores] = getEliteMasks(declineResults, p, featureVars, opt.eliteCount);
[maintainEliteMasks, maintainEliteScores] = getEliteMasks(maintainResults, p, featureVars, opt.eliteCount);

guide = updateGuideBySpecialistElites(guide, declineEliteMasks, maintainEliteMasks, opt);
writetable(guide.table, fullfile(outDir, "Updated_MutationGuide_WithEliteBoost.xlsx"));

fprintf("\nTop Decline parent fitness: %.4f\n", max(declineEliteScores));
fprintf("Top Maintain parent fitness: %.4f\n", max(maintainEliteScores));

%% =========================================================
% 7. Class-aware crossover and child evolution
%% =========================================================

fprintf("\n=== Stage 2: Class-aware crossover child evolution ===\n");

childPop = initializeChildPopulationFromSpecialists( ...
    declineEliteMasks, maintainEliteMasks, p, opt, guide);

[childPop, childLog, childResults] = evolveBalancedChildPopulation( ...
    childPop, declineEliteMasks, maintainEliteMasks, X_task, yBin, DATA_task, ...
    featureVars, featureGroups, guide, splitMode, kfoldDefault, ...
    bayesOpt, nPosteriorSamplesEval, opt);

writetable(childLog, fullfile(outDir, "Child_Balanced_GenerationLog.xlsx"));
writetable(childResults, fullfile(outDir, "Child_Balanced_IndividualResults.xlsx"));

% Export a transparent ranking of unique balanced-child candidates from the
% current rerun. This is a search diagnostic and is intentionally separate
% from the fixed seven-feature manuscript signature used in scripts 05-06.
childCandidateRanking = rankBalancedChildCandidates(childResults, 20);
writetable(childCandidateRanking, ...
    fullfile(outDir, "BalancedChild_CandidateRanking.xlsx"));

fprintf("\n=== Top balanced-child candidates from current rerun ===\n");
disp(childCandidateRanking(1:min(10,height(childCandidateRanking)), :));

%% =========================================================
% 8. Final analysis for best parent and child signatures
%% =========================================================

fprintf("\n=== Stage 3: Final CV posterior/attribution analysis ===\n");

bestDeclineMask = getBestMaskFromResults(declineResults, p, featureVars);
bestMaintainMask = getBestMaskFromResults(maintainResults, p, featureVars);
bestChildMask = getBestMaskFromResults(childResults, p, featureVars);

[declineFinal, declineAttr, declineCoef] = finalCVSignatureAnalysis( ...
    X_task, yBin, DATA_task, bestDeclineMask, featureVars, splitMode, kfoldDefault, ...
    bayesOpt, nPosteriorSamplesFinal, "DeclineParent", 9091);

[maintainFinal, maintainAttr, maintainCoef] = finalCVSignatureAnalysis( ...
    X_task, yBin, DATA_task, bestMaintainMask, featureVars, splitMode, kfoldDefault, ...
    bayesOpt, nPosteriorSamplesFinal, "MaintainParent", 9092);

[childFinal, childAttr, childCoef] = finalCVSignatureAnalysis( ...
    X_task, yBin, DATA_task, bestChildMask, featureVars, splitMode, kfoldDefault, ...
    bayesOpt, nPosteriorSamplesFinal, "BalancedChild", 9093);

writetable(declineAttr, fullfile(outDir, "Final_Attribution_DeclineParent.xlsx"));
writetable(maintainAttr, fullfile(outDir, "Final_Attribution_MaintainParent.xlsx"));
writetable(childAttr, fullfile(outDir, "Final_Attribution_BalancedChild.xlsx"));

writetable(declineCoef, fullfile(outDir, "Final_CoefficientPosterior_DeclineParent.xlsx"));
writetable(maintainCoef, fullfile(outDir, "Final_CoefficientPosterior_MaintainParent.xlsx"));
writetable(childCoef, fullfile(outDir, "Final_CoefficientPosterior_BalancedChild.xlsx"));

finalTable = table();
finalTable = [finalTable; makeFinalSignatureRow("DeclineParent", bestDeclineMask, featureVars, declineFinal)];
finalTable = [finalTable; makeFinalSignatureRow("MaintainParent", bestMaintainMask, featureVars, maintainFinal)];
finalTable = [finalTable; makeFinalSignatureRow("BalancedChild", bestChildMask, featureVars, childFinal)];

writetable(finalTable, fullfile(outDir, "ClassAwareGA_FinalSignatures.xlsx"));

fprintf("\n=== Final signatures ===\n");
disp(finalTable);

fprintf("\nNOTE: The BalancedChild row above is the highest-fitness child from this current rerun.\n");
fprintf("It is not automatically substituted for the fixed seven-feature manuscript signature.\n");
fprintf("See BalancedChild_CandidateRanking.xlsx and README.md for the distinction between\n");
fprintf("discovery reruns and the prespecified signature evaluated by scripts 05-06.\n");

fprintf("\nDone. Results saved to:\n%s\n", outDir);

%% =========================================================
% Local functions
%% =========================================================

function guide = computeGlobalMutationGuide(X, y, featureVars, featureGroups, opt)

    y = double(y(:));
    p = size(X, 2);

    Xz = zscoreWhole(X);

    effect = zeros(p, 1);
    absEffect = zeros(p, 1);
    signEffect = zeros(p, 1);

    for j = 1:p
        x0 = Xz(y == 0, j);
        x1 = Xz(y == 1, j);

        mu0 = mean(x0, "omitnan");
        mu1 = mean(x1, "omitnan");
        v0 = var(x0, 0, "omitnan");
        v1 = var(x1, 0, "omitnan");
        pooled = sqrt(0.5 * (v0 + v1) + eps);

        effect(j) = (mu1 - mu0) / pooled;
        absEffect(j) = abs(effect(j));
        signEffect(j) = sign(effect(j));
    end

    baseScore = normalize01(absEffect);
    declineScore = baseScore .* (0.65 + 0.35 * double(signEffect >= 0));
    maintainScore = baseScore .* (0.65 + 0.35 * double(signEffect <= 0));
    balancedScore = baseScore;

    % Keep only top guide fraction as primary mutation pool, but do not zero out fully.
    cutoff = quantile(baseScore, max(0, 1 - opt.topGuideFraction));
    inGuidePool = baseScore >= cutoff;

    declineScore(~inGuidePool) = 0.25 * declineScore(~inGuidePool);
    maintainScore(~inGuidePool) = 0.25 * maintainScore(~inGuidePool);
    balancedScore(~inGuidePool) = 0.25 * balancedScore(~inGuidePool);

    T = table();
    T.Feature = string(featureVars(:));
    T.FeatureGroup = string(featureGroups(:));
    T.UnivariateEffect_DeclineMinusMaintain = effect;
    T.AbsEffect = absEffect;
    T.BaseGuideScore = baseScore;
    T.DeclineMutationScore = declineScore;
    T.MaintainMutationScore = maintainScore;
    T.BalancedMutationScore = balancedScore;
    T.InGuidePool = inGuidePool;
    T.DeclineEliteFreq = zeros(p, 1);
    T.MaintainEliteFreq = zeros(p, 1);

    guide = struct();
    guide.table = T;
    guide.featureNames = string(featureVars(:));
    guide.featureGroups = string(featureGroups(:));
end

function guide = updateGuideBySpecialistElites(guide, declineEliteMasks, maintainEliteMasks, opt)

    T = guide.table;
    p = height(T);

    if isempty(declineEliteMasks)
        dFreq = zeros(p, 1);
    else
        dFreq = mean(double(declineEliteMasks), 1)';
    end

    if isempty(maintainEliteMasks)
        mFreq = zeros(p, 1);
    else
        mFreq = mean(double(maintainEliteMasks), 1)';
    end

    T.DeclineEliteFreq = dFreq;
    T.MaintainEliteFreq = mFreq;

    T.DeclineMutationScore = normalize01(T.DeclineMutationScore + opt.eliteGuideBoost * dFreq);
    T.MaintainMutationScore = normalize01(T.MaintainMutationScore + opt.eliteGuideBoost * mFreq);
    T.BalancedMutationScore = normalize01(T.BalancedMutationScore + 0.5 * opt.eliteGuideBoost * (dFreq + mFreq));

    guide.table = T;
end

function Xz = zscoreWhole(X)

    mu = mean(X, 1, "omitnan");
    sig = std(X, 0, 1, "omitnan");
    sig(sig < eps) = 1;
    Xz = (X - mu) ./ sig;
    Xz(~isfinite(Xz)) = 0;
end

function x = normalize01(x)

    x = double(x(:));
    mn = min(x, [], "omitnan");
    mx = max(x, [], "omitnan");

    if ~isfinite(mn) || ~isfinite(mx) || abs(mx - mn) < eps
        x = zeros(size(x));
    else
        x = (x - mn) ./ (mx - mn);
    end

    x(~isfinite(x)) = 0;
end

function population = initializePopulation(p, opt, guide, modeName)

    population = false(opt.populationSize, p);

    for i = 1:opt.populationSize
        nFeat = randi([opt.minFeatures, opt.maxFeatures]);
        mask = false(1, p);

        % Half random, half guide-based to keep exploration.
        if rand < 0.50
            idx = randperm(p, min(nFeat, p));
        else
            idx = weightedFeatureSample(mask, guide, string(modeName), nFeat, opt);
        end

        mask(idx) = true;
        population(i, :) = enforceFeatureLimits(mask, opt);
    end
end

function [population, genLogAll, resultAll] = evolveSpecialistPopulation( ...
    population, modeName, X, y, DATA, featureVars, featureGroups, guide, ...
    splitMode, kfoldDefault, bayesOpt, nPosteriorSamples, opt)

    p = size(population, 2);
    resultAll = table();
    genLogAll = table();

    for gen = 1:opt.nSpecialistGenerations

        fprintf("\nSpecialist %s | Generation %d / %d\n", ...
            modeName, gen, opt.nSpecialistGenerations);

        results = table();
        fitnessVec = nan(size(population, 1), 1);

        for i = 1:size(population, 1)
            mask = population(i, :);

            result = evaluateBayesianSubset( ...
                X, y, DATA, mask, featureVars, featureGroups, splitMode, kfoldDefault, ...
                bayesOpt, nPosteriorSamples, modeName, 1000 + gen * 100 + i, opt);

            result.Phase = string(modeName);
            result.Generation = gen;
            result.Individual = i;
            result.SelectedFeatures = strjoin(featureVars(mask), ",");

            results = [results; result];
            fitnessVec(i) = result.Fitness;
        end

        results = movevars(results, ["Phase", "Generation", "Individual"], "Before", 1);
        resultAll = [resultAll; results];

        [bestFit, bestIdx] = max(fitnessVec, [], "omitnan");
        genLog = table();
        genLog.Phase = string(modeName);
        genLog.Generation = gen;
        genLog.BestFitness = bestFit;
        genLog.MeanFitness = mean(fitnessVec, "omitnan");
        genLog.StdFitness = std(fitnessVec, 0, "omitnan");
        genLog.BestNumFeatures = sum(population(bestIdx, :));
        genLog.BestBalancedAccuracy = results.BalancedAccuracy(bestIdx);
        genLog.BestRecall_ClassA = results.Recall_ClassA(bestIdx);
        genLog.BestRecall_ClassB = results.Recall_ClassB(bestIdx);
        genLog.BestAUC = results.AUC(bestIdx);

        genLogAll = [genLogAll; genLog];

        fprintf("Best fitness %.4f | BalAcc %.3f | RecA %.3f | RecB %.3f | AUC %.3f | Features %d\n", ...
            genLog.BestFitness, genLog.BestBalancedAccuracy, genLog.BestRecall_ClassA, ...
            genLog.BestRecall_ClassB, genLog.BestAUC, genLog.BestNumFeatures);

        population = evolveSpecialistStep(population, fitnessVec, guide, modeName, opt);
    end
end

function newPop = evolveSpecialistStep(population, fitnessVec, guide, modeName, opt)

    [popSize, ~] = size(population);
    newPop = false(size(population));

    [~, order] = sort(fitnessVec, "descend", "MissingPlacement", "last");
    eliteCount = min(opt.eliteCount, popSize);
    newPop(1:eliteCount, :) = population(order(1:eliteCount), :);

    for i = eliteCount+1:popSize
        pA = tournamentSelect(population, fitnessVec, opt.tournamentSize);
        pB = tournamentSelect(population, fitnessVec, opt.tournamentSize);
        child = uniformCrossover(pA, pB);
        child = guidedMutate(child, guide, string(modeName), opt);
        child = enforceFeatureLimits(child, opt);
        newPop(i, :) = child;
    end
end

function childPop = initializeChildPopulationFromSpecialists(declineEliteMasks, maintainEliteMasks, p, opt, guide)

    childPop = false(opt.populationSize, p);

    if isempty(declineEliteMasks) || isempty(maintainEliteMasks)
        childPop = initializePopulation(p, opt, guide, "Balanced");
        return;
    end

    for i = 1:opt.populationSize
        d = declineEliteMasks(randi(size(declineEliteMasks, 1)), :);
        m = maintainEliteMasks(randi(size(maintainEliteMasks, 1)), :);
        child = classAwareCrossover(d, m);
        child = guidedMutate(child, guide, "Balanced", opt);
        childPop(i, :) = enforceFeatureLimits(child, opt);
    end
end

function [population, genLogAll, resultAll] = evolveBalancedChildPopulation( ...
    population, declineEliteMasks, maintainEliteMasks, X, y, DATA, featureVars, featureGroups, guide, ...
    splitMode, kfoldDefault, bayesOpt, nPosteriorSamples, opt)

    p = size(population, 2);
    resultAll = table();
    genLogAll = table();

    for gen = 1:opt.nChildGenerations

        fprintf("\nBalanced child | Generation %d / %d\n", gen, opt.nChildGenerations);

        results = table();
        fitnessVec = nan(size(population, 1), 1);

        for i = 1:size(population, 1)
            mask = population(i, :);

            result = evaluateBayesianSubset( ...
                X, y, DATA, mask, featureVars, featureGroups, splitMode, kfoldDefault, ...
                bayesOpt, nPosteriorSamples, "BalancedChild", 3000 + gen * 100 + i, opt);

            result.Phase = "BalancedChild";
            result.Generation = gen;
            result.Individual = i;
            result.SelectedFeatures = strjoin(featureVars(mask), ",");

            results = [results; result];
            fitnessVec(i) = result.Fitness;
        end

        results = movevars(results, ["Phase", "Generation", "Individual"], "Before", 1);
        resultAll = [resultAll; results];

        [bestFit, bestIdx] = max(fitnessVec, [], "omitnan");
        genLog = table();
        genLog.Phase = "BalancedChild";
        genLog.Generation = gen;
        genLog.BestFitness = bestFit;
        genLog.MeanFitness = mean(fitnessVec, "omitnan");
        genLog.StdFitness = std(fitnessVec, 0, "omitnan");
        genLog.BestNumFeatures = sum(population(bestIdx, :));
        genLog.BestBalancedAccuracy = results.BalancedAccuracy(bestIdx);
        genLog.BestRecall_ClassA = results.Recall_ClassA(bestIdx);
        genLog.BestRecall_ClassB = results.Recall_ClassB(bestIdx);
        genLog.BestAUC = results.AUC(bestIdx);

        genLogAll = [genLogAll; genLog];

        fprintf("Best fitness %.4f | BalAcc %.3f | RecA %.3f | RecB %.3f | AUC %.3f | Features %d\n", ...
            genLog.BestFitness, genLog.BestBalancedAccuracy, genLog.BestRecall_ClassA, ...
            genLog.BestRecall_ClassB, genLog.BestAUC, genLog.BestNumFeatures);

        population = evolveChildStep(population, fitnessVec, declineEliteMasks, maintainEliteMasks, guide, opt);
    end
end

function newPop = evolveChildStep(population, fitnessVec, declineEliteMasks, maintainEliteMasks, guide, opt)

    [popSize, p] = size(population);
    newPop = false(size(population));

    [~, order] = sort(fitnessVec, "descend", "MissingPlacement", "last");
    eliteCount = min(opt.eliteCount, popSize);
    newPop(1:eliteCount, :) = population(order(1:eliteCount), :);

    for i = eliteCount+1:popSize

        if ~isempty(declineEliteMasks) && ~isempty(maintainEliteMasks) && rand < 0.75
            d = declineEliteMasks(randi(size(declineEliteMasks, 1)), :);
            m = maintainEliteMasks(randi(size(maintainEliteMasks, 1)), :);
            child = classAwareCrossover(d, m);

            % Occasionally mix with a strong previous child.
            if rand < 0.35
                pChild = tournamentSelect(population, fitnessVec, opt.tournamentSize);
                child = uniformCrossover(child, pChild);
            end
        else
            pA = tournamentSelect(population, fitnessVec, opt.tournamentSize);
            pB = tournamentSelect(population, fitnessVec, opt.tournamentSize);
            child = uniformCrossover(pA, pB);
        end

        % Balanced mutation alternates between adding decline-like and maintain-like genes.
        r = rand;
        if r < 0.35
            mode = "Decline";
        elseif r < 0.70
            mode = "Maintain";
        else
            mode = "Balanced";
        end

        child = guidedMutate(child, guide, mode, opt);
        child = enforceFeatureLimits(child, opt);
        newPop(i, :) = child;
    end
end

function child = classAwareCrossover(declineParent, maintainParent)

    child = false(size(declineParent));

    common = declineParent & maintainParent;
    onlyD = declineParent & ~maintainParent;
    onlyM = maintainParent & ~declineParent;

    child(common) = true;

    dIdx = find(onlyD);
    mIdx = find(onlyM);

    if ~isempty(dIdx)
        child(dIdx(rand(size(dIdx)) < 0.55)) = true;
    end

    if ~isempty(mIdx)
        child(mIdx(rand(size(mIdx)) < 0.55)) = true;
    end
end

function child = uniformCrossover(parentA, parentB)

    mask = rand(size(parentA)) < 0.5;
    child = parentA;
    child(mask) = parentB(mask);
end

function child = guidedMutate(child, guide, modeName, opt)

    p = numel(child);

    % Small random flips for exploration.
    flipMask = rand(1, p) < opt.randomFlipRate;
    child(flipMask) = ~child(flipMask);

    if rand > opt.guidedMutationRate
        return;
    end

    r = rand;
    if r < opt.addProb
        addIdx = weightedFeatureSample(child, guide, modeName, 1, opt);
        child(addIdx) = true;
    elseif r < opt.addProb + opt.dropProb
        dropIdx = guidedDropSample(child, guide, modeName, 1);
        child(dropIdx) = false;
    else
        dropIdx = guidedDropSample(child, guide, modeName, 1);
        addIdx = weightedFeatureSample(child, guide, modeName, 1, opt);
        child(dropIdx) = false;
        child(addIdx) = true;
    end
end

function idx = weightedFeatureSample(currentMask, guide, modeName, nSample, opt)

    T = guide.table;
    p = height(T);

    switch string(modeName)
        case {"Decline", "DeclineParent"}
            score = T.DeclineMutationScore;
        case {"Maintain", "MaintainParent"}
            score = T.MaintainMutationScore;
        otherwise
            score = T.BalancedMutationScore;
    end

    score = double(score(:));
    score(currentMask(:)) = 0;

    % Group diversity bonus: prefer feature groups not already selected.
    if any(currentMask)
        currentGroups = guide.featureGroups(currentMask(:));
        for j = 1:p
            if ~currentMask(j) && ~ismember(guide.featureGroups(j), currentGroups)
                score(j) = score(j) * (1 + opt.groupDiversityWeight);
            end
        end
    end

    if all(score <= 0) || all(~isfinite(score))
        candidate = find(~currentMask(:));
        if isempty(candidate)
            idx = [];
        else
            nSample = min(nSample, numel(candidate));
            idx = candidate(randperm(numel(candidate), nSample));
        end
        return;
    end

    score(~isfinite(score)) = 0;
    score = score + 1e-12;

    idx = zeros(nSample, 1);
    available = true(p, 1);
    available(currentMask(:)) = false;

    for k = 1:nSample
        w = score;
        w(~available) = 0;

        if sum(w) <= 0
            cand = find(available);
            if isempty(cand)
                idx = idx(1:k-1);
                return;
            end
            chosen = cand(randi(numel(cand)));
        else
            chosen = weightedChoice(w);
        end

        idx(k) = chosen;
        available(chosen) = false;
    end
end

function idx = guidedDropSample(currentMask, guide, modeName, nSample)

    selected = find(currentMask(:));
    if isempty(selected)
        idx = [];
        return;
    end

    T = guide.table;
    switch string(modeName)
        case {"Decline", "DeclineParent"}
            keepScore = T.DeclineMutationScore;
        case {"Maintain", "MaintainParent"}
            keepScore = T.MaintainMutationScore;
        otherwise
            keepScore = T.BalancedMutationScore;
    end

    keepScore = double(keepScore(:));
    dropWeight = 1 ./ (keepScore(selected) + 1e-6);
    dropWeight(~isfinite(dropWeight)) = 1;

    nSample = min(nSample, numel(selected));
    idx = zeros(nSample, 1);
    available = true(numel(selected), 1);

    for k = 1:nSample
        w = dropWeight;
        w(~available) = 0;
        if sum(w) <= 0
            avail = find(available);
            pickLocal = avail(randi(numel(avail)));
        else
            pickLocal = weightedChoice(w);
        end
        idx(k) = selected(pickLocal);
        available(pickLocal) = false;
    end
end

function chosen = weightedChoice(w)

    w = double(w(:));
    w(~isfinite(w) | w < 0) = 0;
    s = sum(w);
    if s <= 0
        candidate = find(w >= 0);
        chosen = candidate(randi(numel(candidate)));
        return;
    end
    c = cumsum(w / s);
    r = rand;
    chosen = find(c >= r, 1, "first");
end

function parent = tournamentSelect(population, fitnessVec, tournamentSize)

    n = size(population, 1);
    idx = randi(n, tournamentSize, 1);
    [~, bestLocal] = max(fitnessVec(idx));
    parent = population(idx(bestLocal), :);
end

function child = enforceFeatureLimits(child, opt)

    active = find(child);
    inactive = find(~child);

    if numel(active) > opt.maxFeatures
        dropN = numel(active) - opt.maxFeatures;
        dropIdx = active(randperm(numel(active), dropN));
        child(dropIdx) = false;
    end

    active = find(child);
    inactive = find(~child);

    if numel(active) < opt.minFeatures && ~isempty(inactive)
        addN = min(opt.minFeatures - numel(active), numel(inactive));
        addIdx = inactive(randperm(numel(inactive), addN));
        child(addIdx) = true;
    end
end

function result = evaluateBayesianSubset( ...
    X, y, DATA, mask, featureVars, featureGroups, splitMode, kfoldDefault, ...
    bayesOpt, nPosteriorSamples, modeName, seed, opt)

    selectedIdx = find(mask);
    nFeat = numel(selectedIdx);

    if nFeat < 2
        result = makeBadResult(nFeat);
        return;
    end

    folds = makeCVFolds(DATA, y, splitMode, kfoldDefault, seed);

    allY = [];
    allPred = [];
    allScore = [];
    allProbStd = [];
    allEntropy = [];
    allAttrUnc = [];
    allAttrUncA = [];
    allAttrUncB = [];

    for f = 1:numel(folds)

        tr = folds(f).train;
        te = folds(f).test;

        Xtr = X(tr, selectedIdx);
        Xte = X(te, selectedIdx);
        ytr = y(tr);
        yte = y(te);

        if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2
            continue;
        end

        [XtrZ, XteZ] = zscoreTrainTest(Xtr, Xte);
        wtr = makeClassWeights(ytr);

        mdl = trainBayesianLogisticLaplaceWeighted(XtrZ, ytr, wtr, bayesOpt);
        pred = predictBayesianLogisticWithSamples(mdl, XteZ, nPosteriorSamples);

        yhat = double(pred.probMean >= 0.5);

        [attrUnc, attrUncA, attrUncB] = quickAttributionUncertainty( ...
            XteZ, yte, pred.probSamples, pred.betaSamples);

        allY = [allY; yte(:)];
        allPred = [allPred; yhat(:)];
        allScore = [allScore; pred.probMean(:)];
        allProbStd = [allProbStd; pred.probStd(:)];
        allEntropy = [allEntropy; pred.entropy(:)];
        allAttrUnc = [allAttrUnc; attrUnc];
        allAttrUncA = [allAttrUncA; attrUncA];
        allAttrUncB = [allAttrUncB; attrUncB];
    end

    if isempty(allY)
        result = makeBadResult(nFeat);
        return;
    end

    metric = computeBinaryMetrics(allY, allPred, allScore);

    result = metric;
    result.NumFeatures = nFeat;
    result.MeanProbStd = mean(allProbStd, "omitnan");
    result.MeanEntropy = mean(allEntropy, "omitnan");
    result.MeanAttributionUncertainty = mean(allAttrUnc, "omitnan");
    result.MeanAttributionUncertainty_ClassA = mean(allAttrUncA, "omitnan");
    result.MeanAttributionUncertainty_ClassB = mean(allAttrUncB, "omitnan");
    result.MeanProbStd_ClassA = mean(allProbStd(allY == 0), "omitnan");
    result.MeanProbStd_ClassB = mean(allProbStd(allY == 1), "omitnan");

    result.Redundancy = computeMeanAbsCorr(X(:, selectedIdx));
    result.Fitness = computeClassAwareFitness(result, modeName, opt);
end

function bad = makeBadResult(nFeat)

    bad = table();
    bad.Accuracy = 0;
    bad.BalancedAccuracy = 0;
    bad.MacroF1 = 0;
    bad.Precision_ClassA = 0;
    bad.Precision_ClassB = 0;
    bad.Recall_ClassA = 0;
    bad.Recall_ClassB = 0;
    bad.Specificity = 0;
    bad.AUC = 0.5;
    bad.AUPRC = 0;
    bad.BrierScore = 1;
    bad.TP = 0;
    bad.TN = 0;
    bad.FP = 0;
    bad.FN = 0;
    bad.NumFeatures = nFeat;
    bad.MeanProbStd = nan;
    bad.MeanEntropy = nan;
    bad.MeanAttributionUncertainty = nan;
    bad.MeanAttributionUncertainty_ClassA = nan;
    bad.MeanAttributionUncertainty_ClassB = nan;
    bad.MeanProbStd_ClassA = nan;
    bad.MeanProbStd_ClassB = nan;
    bad.Redundancy = nan;
    bad.Fitness = -inf;
end

function fitness = computeClassAwareFitness(m, modeName, opt)

    aucVal = m.AUC;
    if isnan(aucVal)
        aucVal = 0.5;
    end

    recA = m.Recall_ClassA;
    recB = m.Recall_ClassB;
    minRec = min(recA, recB);

    switch string(modeName)
        case "DeclineParent"
            base = 0.45 * recB + 0.20 * aucVal + 0.20 * m.BalancedAccuracy + 0.15 * m.MacroF1;
            oppositePenalty = 0.25 * max(0, opt.minOppositeRecall - recA);
            probPenalty = 0.08 * min(m.MeanProbStd_ClassB / opt.probStdScale, 2);
            attrPenalty = 0.07 * min(m.MeanAttributionUncertainty_ClassB / opt.attrUncScale, 2);

        case "MaintainParent"
            base = 0.45 * recA + 0.20 * aucVal + 0.20 * m.BalancedAccuracy + 0.15 * m.MacroF1;
            oppositePenalty = 0.25 * max(0, opt.minOppositeRecall - recB);
            probPenalty = 0.08 * min(m.MeanProbStd_ClassA / opt.probStdScale, 2);
            attrPenalty = 0.07 * min(m.MeanAttributionUncertainty_ClassA / opt.attrUncScale, 2);

        otherwise % BalancedChild
            base = 0.35 * m.BalancedAccuracy + 0.25 * aucVal + 0.20 * m.MacroF1 + 0.20 * minRec;
            oppositePenalty = 0;
            probPenalty = 0.08 * min(m.MeanProbStd / opt.probStdScale, 2);
            attrPenalty = 0.07 * min(m.MeanAttributionUncertainty / opt.attrUncScale, 2);
    end

    featurePenalty = 0.025 * min(m.NumFeatures / opt.maxFeatures, 2);
    redundancyPenalty = 0.035 * min(m.Redundancy / opt.redundancyScale, 2);

    fitness = base - oppositePenalty - probPenalty - attrPenalty - featurePenalty - redundancyPenalty;
end

function r = computeMeanAbsCorr(X)

    if size(X, 2) < 2
        r = 0;
        return;
    end

    C = corr(X, "Rows", "pairwise");
    C(~isfinite(C)) = 0;
    C(1:size(C,1)+1:end) = nan;
    r = mean(abs(C(:)), "omitnan");
    if ~isfinite(r)
        r = 0;
    end
end

function [attrUnc, attrUncA, attrUncB] = quickAttributionUncertainty(Xte, yte, probSamples, betaSamples)

    nTest = size(Xte, 1);
    p = size(Xte, 2);
    S = size(probSamples, 2);

    probSlope = probSamples .* (1 - probSamples);

    sampleFeatureUnc = nan(nTest, p);

    for j = 1:p
        betaJ = betaSamples(:, j + 1)';
        attrJ = probSlope .* repmat(betaJ, nTest, 1) .* repmat(Xte(:, j), 1, S);
        sampleFeatureUnc(:, j) = std(attrJ, 0, 2, "omitnan");
    end

    sampleUnc = mean(sampleFeatureUnc, 2, "omitnan");

    attrUnc = mean(sampleUnc, "omitnan");
    attrUncA = mean(sampleUnc(yte == 0), "omitnan");
    attrUncB = mean(sampleUnc(yte == 1), "omitnan");
end

function [eliteMasks, eliteScores] = getEliteMasks(resultTable, p, featureVars, eliteCount)

    if isempty(resultTable)
        eliteMasks = false(0, p);
        eliteScores = [];
        return;
    end

    [~, order] = sort(resultTable.Fitness, "descend", "MissingPlacement", "last");
    order = order(1:min(eliteCount, numel(order)));

    eliteMasks = false(numel(order), p);
    eliteScores = resultTable.Fitness(order);

    for i = 1:numel(order)
        selected = string(split(string(resultTable.SelectedFeatures(order(i))), ","));
        selected = selected(selected ~= "");
        eliteMasks(i, :) = ismember(string(featureVars(:))', selected);
    end
end

function ranked = rankBalancedChildCandidates(resultTable, topN)
% Rank unique balanced-child signatures from the current search run.
%
% Primary ordering:
%   1) higher balanced accuracy
%   2) higher AUC
%   3) higher decline recall
%   4) higher fitness
%   5) fewer selected features
%
% This ranking is a search diagnostic. It does not redefine the fixed
% manuscript signature used in scripts 05 and 06.

    if nargin < 2 || isempty(topN)
        topN = inf;
    end

    if isempty(resultTable)
        ranked = table();
        return;
    end

    sig = string(resultTable.SelectedFeatures);
    [uniqueSig, ~, groupIdx] = unique(sig, "stable");

    bestRows = zeros(numel(uniqueSig), 1);

    for g = 1:numel(uniqueSig)
        idx = find(groupIdx == g);

        key = [ ...
            -double(resultTable.BalancedAccuracy(idx)), ...
            -double(resultTable.AUC(idx)), ...
            -double(resultTable.Recall_ClassB(idx)), ...
            -double(resultTable.Fitness(idx)), ...
             double(resultTable.NumFeatures(idx))];

        [~, ord] = sortrows(key, 1:size(key,2));
        bestRows(g) = idx(ord(1));
    end

    ranked = resultTable(bestRows, :);

    key = [ ...
        -double(ranked.BalancedAccuracy), ...
        -double(ranked.AUC), ...
        -double(ranked.Recall_ClassB), ...
        -double(ranked.Fitness), ...
         double(ranked.NumFeatures)];

    [~, ord] = sortrows(key, 1:size(key,2));
    ranked = ranked(ord, :);

    ranked.CandidateRank_ByBalancedAccuracy = (1:height(ranked))';
    ranked = movevars(ranked, "CandidateRank_ByBalancedAccuracy", "Before", 1);

    if isfinite(topN) && height(ranked) > topN
        ranked = ranked(1:topN, :);
    end
end

function mask = getBestMaskFromResults(resultTable, p, featureVars)

    [~, idx] = max(resultTable.Fitness, [], "omitnan");
    selected = string(split(string(resultTable.SelectedFeatures(idx)), ","));
    selected = selected(selected ~= "");
    mask = ismember(string(featureVars(:))', selected);

    if ~any(mask)
        mask = false(1, p);
    end
end

function row = makeFinalSignatureRow(label, mask, featureVars, metrics)

    row = table();
    row.Signature = string(label);
    row.NumFeatures = sum(mask);
    row.SelectedFeatures = strjoin(featureVars(mask), ",");
    row.Accuracy = metrics.Accuracy;
    row.BalancedAccuracy = metrics.BalancedAccuracy;
    row.MacroF1 = metrics.MacroF1;
    row.Recall_Maintain = metrics.Recall_ClassA;
    row.Recall_Decline = metrics.Recall_ClassB;
    row.AUC = metrics.AUC;
    row.AUPRC = metrics.AUPRC;
    row.MeanProbStd = metrics.MeanProbStd;
    row.MeanEntropy = metrics.MeanEntropy;
    row.MeanAttributionUncertainty = metrics.MeanAttributionUncertainty;
end

function [finalMetrics, attrSummary, coefSummary] = finalCVSignatureAnalysis( ...
    X, y, DATA, mask, featureVars, splitMode, kfoldDefault, bayesOpt, nPosteriorSamples, targetMode, seed)

    selectedIdx = find(mask);
    selectedFeatureNames = string(featureVars(selectedIdx));

    folds = makeCVFolds(DATA, y, splitMode, kfoldDefault, seed);

    allY = [];
    allPred = [];
    allScore = [];
    allProbStd = [];
    allEntropy = [];
    allAttrUnc = [];

    allAttrRows = table();
    allCoefRows = table();

    for f = 1:numel(folds)

        tr = folds(f).train;
        te = folds(f).test;

        Xtr = X(tr, selectedIdx);
        Xte = X(te, selectedIdx);
        ytr = y(tr);
        yte = y(te);

        if numel(unique(ytr)) < 2 || numel(unique(yte)) < 2
            continue;
        end

        [XtrZ, XteZ] = zscoreTrainTest(Xtr, Xte);
        wtr = makeClassWeights(ytr);

        mdl = trainBayesianLogisticLaplaceWeighted(XtrZ, ytr, wtr, bayesOpt);
        pred = predictBayesianLogisticWithSamples(mdl, XteZ, nPosteriorSamples);

        yhat = double(pred.probMean >= 0.5);

        [attrUnc, ~, ~] = quickAttributionUncertainty(XteZ, yte, pred.probSamples, pred.betaSamples);

        allY = [allY; yte(:)];
        allPred = [allPred; yhat(:)];
        allScore = [allScore; pred.probMean(:)];
        allProbStd = [allProbStd; pred.probStd(:)];
        allEntropy = [allEntropy; pred.entropy(:)];
        allAttrUnc = [allAttrUnc; attrUnc];

        attrFold = computeBayesianAttributionFoldSummary( ...
            XteZ, yte, pred.probSamples, pred.betaSamples, selectedFeatureNames);
        attrFold.Fold = repmat(f, height(attrFold), 1);
        attrFold.Signature = repmat(string(targetMode), height(attrFold), 1);
        allAttrRows = [allAttrRows; attrFold];

        coefFold = summarizeCoefficientPosterior(mdl, selectedFeatureNames);
        coefFold.Fold = repmat(f, height(coefFold), 1);
        coefFold.Signature = repmat(string(targetMode), height(coefFold), 1);
        allCoefRows = [allCoefRows; coefFold];
    end

    finalMetrics = computeBinaryMetrics(allY, allPred, allScore);
    finalMetrics.MeanProbStd = mean(allProbStd, "omitnan");
    finalMetrics.MeanEntropy = mean(allEntropy, "omitnan");
    finalMetrics.MeanAttributionUncertainty = mean(allAttrUnc, "omitnan");

    attrSummary = summarizeAttributionAcrossFolds(allAttrRows);
    coefSummary = summarizeCoefficientAcrossFolds(allCoefRows);
end

function attrT = computeBayesianAttributionFoldSummary(Xte, yte, probSamples, betaSamples, featureNames)

    nTest = size(Xte, 1);
    p = size(Xte, 2);
    S = size(probSamples, 2);
    probSlope = probSamples .* (1 - probSamples);

    attrT = table();

    for j = 1:p
        betaJ = betaSamples(:, j + 1)';
        attrJ = probSlope .* repmat(betaJ, nTest, 1) .* repmat(Xte(:, j), 1, S);

        sampleMeanAttr = mean(attrJ, 2, "omitnan");
        sampleStdAttr = std(attrJ, 0, 2, "omitnan");
        sampleMeanAbsAttr = mean(abs(attrJ), 2, "omitnan");

        vals = attrJ(:);
        vals = vals(isfinite(vals));

        if isempty(vals)
            ciLow = nan; ciHigh = nan; posProb = nan;
        else
            ci = prctile(vals, [2.5, 97.5]);
            ciLow = ci(1); ciHigh = ci(2); posProb = mean(vals > 0);
        end

        row = table();
        row.Feature = string(featureNames(j));
        row.FeatureGroup = inferFeatureGroup(string(featureNames(j)));
        row.MeanSignedAttribution = mean(sampleMeanAttr, "omitnan");
        row.MeanAbsAttribution = mean(sampleMeanAbsAttr, "omitnan");
        row.MeanAttributionUncertainty = mean(sampleStdAttr, "omitnan");
        row.PositiveAttributionProb = posProb;
        row.AttributionCI025 = ciLow;
        row.AttributionCI975 = ciHigh;
        row.MeanSignedAttribution_ClassA = mean(sampleMeanAttr(yte == 0), "omitnan");
        row.MeanSignedAttribution_ClassB = mean(sampleMeanAttr(yte == 1), "omitnan");

        attrT = [attrT; row];
    end
end

function summaryT = summarizeAttributionAcrossFolds(allAttrRows)

    if isempty(allAttrRows)
        summaryT = table();
        return;
    end

    groupVars = ["Signature", "Feature", "FeatureGroup"];
    metricVars = [
        "MeanSignedAttribution"
        "MeanAbsAttribution"
        "MeanAttributionUncertainty"
        "PositiveAttributionProb"
        "MeanSignedAttribution_ClassA"
        "MeanSignedAttribution_ClassB"
    ];

    summaryT = groupsummary(allAttrRows, groupVars, {"mean", "std"}, metricVars);
    summaryT.AttributionStabilityScore = summaryT.mean_MeanAbsAttribution ./ ...
        (summaryT.mean_MeanAttributionUncertainty + eps);

    direction = strings(height(summaryT), 1);
    direction(summaryT.mean_MeanSignedAttribution > 0) = "Toward_Decline_to_DEM";
    direction(summaryT.mean_MeanSignedAttribution < 0) = "Toward_Maintain_CN_SCD";
    direction(summaryT.mean_MeanSignedAttribution == 0) = "Neutral";
    summaryT.AttributionDirection = direction;

    summaryT = sortrows(summaryT, ["AttributionStabilityScore", "mean_MeanAbsAttribution"], ["descend", "descend"]);
end

function coefT = summarizeCoefficientPosterior(mdl, featureVars)

    betaMean = mdl.betaMAP;
    betaStd = sqrt(max(diag(mdl.covBeta), 0));
    lower = betaMean - 1.96 * betaStd;
    upper = betaMean + 1.96 * betaStd;

    featureNames = ["Intercept"; string(featureVars(:))];
    probPositive = nan(numel(betaMean), 1);

    for j = 1:numel(betaMean)
        if betaStd(j) > 0
            probPositive(j) = 1 - normcdf(0, betaMean(j), betaStd(j));
        else
            probPositive(j) = double(betaMean(j) > 0);
        end
    end

    coefT = table();
    coefT.Feature = featureNames;
    coefT.BetaMean = betaMean;
    coefT.BetaStd = betaStd;
    coefT.BetaCI025 = lower;
    coefT.BetaCI975 = upper;
    coefT.ProbPositiveEffect = probPositive;
    coefT.AbsZ = abs(betaMean ./ max(betaStd, eps));
end

function summaryT = summarizeCoefficientAcrossFolds(allCoefRows)

    if isempty(allCoefRows)
        summaryT = table();
        return;
    end

    groupVars = ["Signature", "Feature"];
    metricVars = ["BetaMean", "BetaStd", "ProbPositiveEffect", "AbsZ"];

    summaryT = groupsummary(allCoefRows, groupVars, {"mean", "std"}, metricVars);
    if ismember("mean_AbsZ", string(summaryT.Properties.VariableNames))
        summaryT = sortrows(summaryT, "mean_AbsZ", "descend");
    end
end

function folds = makeCVFolds(DATA, yBin, splitMode, kfoldDefault, seed)

    rng(seed);
    yBin = double(yBin(:));

    if splitMode == "FileLevel"
        minN = min(sum(yBin == 0), sum(yBin == 1));
        kfold = min(kfoldDefault, minN);
        cv = cvpartition(categorical(yBin), "KFold", kfold);
        folds = struct("train", {}, "test", {});
        for f = 1:cv.NumTestSets
            folds(f).train = training(cv, f);
            folds(f).test = test(cv, f);
        end

    elseif splitMode == "SubjectLevel"
        if ~ismember("ObjectID", string(DATA.Properties.VariableNames))
            error("SubjectLevel split requires ObjectID column.");
        end

        subj = string(DATA.ObjectID);
        [uSubj, ~, g] = unique(subj, "stable");
        nSubj = numel(uSubj);
        subjLabel = zeros(nSubj, 1);

        for i = 1:nSubj
            yi = yBin(g == i);
            if any(yi == 1)
                subjLabel(i) = 1;
            else
                subjLabel(i) = 0;
            end
        end

        minGroupN = min(sum(subjLabel == 0), sum(subjLabel == 1));
        kfold = min(kfoldDefault, minGroupN);
        if kfold < 2
            error("Not enough subjects per class for SubjectLevel KFold.");
        end

        foldIdBySubj = stratifiedSubjectFoldAssignment(subjLabel, kfold);
        folds = struct("train", {}, "test", {});
        for f = 1:kfold
            testSubjIdx = find(foldIdBySubj == f);
            testMask = ismember(g, testSubjIdx);
            trainMask = ~testMask;
            folds(f).train = trainMask;
            folds(f).test = testMask;
        end
    else
        error("Unknown splitMode: %s", splitMode);
    end
end

function foldId = stratifiedSubjectFoldAssignment(label, kfold)

    label = double(label(:));
    n = numel(label);
    foldId = zeros(n, 1);

    for c = [0, 1]
        idx = find(label == c);
        idx = idx(randperm(numel(idx)));
        for i = 1:numel(idx)
            foldId(idx(i)) = mod(i - 1, kfold) + 1;
        end
    end
end

function [XtrZ, XteZ] = zscoreTrainTest(Xtr, Xte)

    mu = mean(Xtr, 1, "omitnan");
    sig = std(Xtr, 0, 1, "omitnan");
    sig(sig < eps) = 1;
    XtrZ = (Xtr - mu) ./ sig;
    XteZ = (Xte - mu) ./ sig;
    XtrZ(~isfinite(XtrZ)) = 0;
    XteZ(~isfinite(XteZ)) = 0;
end

function w = makeClassWeights(y)

    y = double(y(:));
    n = numel(y);
    n0 = sum(y == 0);
    n1 = sum(y == 1);
    w = ones(n, 1);
    w(y == 0) = n / max(2 * n0, 1);
    w(y == 1) = n / max(2 * n1, 1);
    w = w / mean(w);
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

function pred = predictBayesianLogisticWithSamples(mdl, X, nSamples)

    n = size(X, 1);
    Xb = [ones(n, 1), X];
    betaSamples = sampleMVN(mdl.betaMAP(:)', mdl.covBeta, nSamples);
    probSamples = sigmoidStable(Xb * betaSamples');

    probMean = mean(probSamples, 2);
    probStd = std(probSamples, 0, 2);
    ci = prctile(probSamples, [2.5, 97.5], 2);

    pred = struct();
    pred.probMean = probMean;
    pred.probStd = probStd;
    pred.probCILower = ci(:, 1);
    pred.probCIUpper = ci(:, 2);
    pred.entropy = binaryEntropy(probMean);
    pred.probSamples = probSamples;
    pred.betaSamples = betaSamples;
end

function metric = computeBinaryMetrics(yTrue, yPred, scorePos)

    yTrue = double(yTrue(:));
    yPred = double(yPred(:));
    scorePos = double(scorePos(:));

    acc = mean(yTrue == yPred);
    tp = sum(yTrue == 1 & yPred == 1);
    tn = sum(yTrue == 0 & yPred == 0);
    fp = sum(yTrue == 0 & yPred == 1);
    fn = sum(yTrue == 1 & yPred == 0);

    recall0 = tn / max(tn + fp, 1);
    recall1 = tp / max(tp + fn, 1);
    precision0 = tn / max(tn + fn, 1);
    precision1 = tp / max(tp + fp, 1);
    f10 = 2 * precision0 * recall0 / max(precision0 + recall0, eps);
    f11 = 2 * precision1 * recall1 / max(precision1 + recall1, eps);

    balAcc = mean([recall0, recall1]);
    macroF1 = mean([f10, f11]);
    aucVal = simpleAUC(scorePos, yTrue, 1);
    auprcVal = simpleAUPRC(scorePos, yTrue, 1);
    score01 = min(max(scorePos, 0), 1);
    brier = mean((score01 - yTrue).^2, "omitnan");

    metric = table();
    metric.Accuracy = acc;
    metric.BalancedAccuracy = balAcc;
    metric.MacroF1 = macroF1;
    metric.Precision_ClassA = precision0;
    metric.Precision_ClassB = precision1;
    metric.Recall_ClassA = recall0;
    metric.Recall_ClassB = recall1;
    metric.Specificity = recall0;
    metric.AUC = aucVal;
    metric.AUPRC = auprcVal;
    metric.BrierScore = brier;
    metric.TP = tp;
    metric.TN = tn;
    metric.FP = fp;
    metric.FN = fn;
end

function aucVal = simpleAUC(score, y, positiveClass)

    score = double(score(:));
    y = double(y(:));
    valid = isfinite(score) & isfinite(y);
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
    aucVal = (sum(r(pos)) - nPos * (nPos + 1) / 2) / (nPos * nNeg + eps);
end

function auprc = simpleAUPRC(score, y, positiveClass)

    score = double(score(:));
    y = double(y(:));
    valid = isfinite(score) & isfinite(y);
    score = score(valid);
    y = y(valid);
    pos = y == positiveClass;
    if sum(pos) < 1
        auprc = nan;
        return;
    end
    [~, order] = sort(score, "descend");
    ySorted = y(order);
    tp = cumsum(ySorted == positiveClass);
    fp = cumsum(ySorted ~= positiveClass);
    precision = tp ./ max(tp + fp, 1);
    recall = tp ./ max(sum(pos), 1);
    recall = [0; recall];
    precision = [1; precision];
    auprc = trapz(recall, precision);
end

function X = sampleMVN(mu, C, n)

    mu = double(mu(:)');
    p = numel(mu);
    C = (C + C') / 2;
    jitter = 1e-8;
    success = false;
    for t = 1:8
        [L, flag] = chol(C + jitter * eye(p), "lower");
        if flag == 0
            success = true;
            break;
        end
        jitter = jitter * 10;
    end
    if ~success
        [V, D] = eig(C);
        eigVals = max(diag(D), 1e-8);
        L = V * diag(sqrt(eigVals));
    end
    X = mu + randn(n, p) * L';
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

function group = inferFeatureGroup(featureName)

    f = lower(string(featureName));

    if contains(f, "theta_alpha") || contains(f, "thetaalpha")
        group = "ThetaAlphaRatio";
    elseif contains(f, "delta_alpha") || contains(f, "deltaalpha")
        group = "DeltaAlphaRatio";
    elseif contains(f, "theta_beta") || contains(f, "thetabeta")
        group = "ThetaBetaRatio";
    elseif contains(f, "delta_beta") || contains(f, "deltabeta")
        group = "DeltaBetaRatio";
    elseif contains(f, "median_freq") || contains(f, "medianfreq")
        group = "MedianFrequency";
    elseif contains(f, "sef95") || contains(f, "sef")
        group = "SEF";
    elseif contains(f, "corr")
        group = "Connectivity";
    elseif contains(f, "alpha")
        group = "Alpha";
    elseif contains(f, "theta")
        group = "Theta";
    elseif contains(f, "delta")
        group = "Delta";
    elseif contains(f, "beta")
        group = "Beta";
    elseif contains(f, "gamma")
        group = "Gamma";
    else
        group = "Other";
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
    [G, names] = findgroups(x);
    counts = splitapply(@numel, x, G);
    summaryTable = table(names, counts, 'VariableNames', {char(varName), 'Count'});
end
