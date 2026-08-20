%% 03_compare_ml_and_bayesian_models.m
% Exploratory preprocessing, feature-count, and classifier benchmark for BUESA.
% BUESA benchmark experiment:
% - Compare preprocessing strategies across ML and Bayesian logistic regression
% - Compare all features vs. training-only top-K features
% - Save raw metrics, summary tables, selected features, and simple figures
%
% Leakage-control rule:
% Every preprocessing statistic, feature ranking, top-K selection, class weight,
% and threshold is estimated from the training fold only.
% The held-out test fold is used only once for final evaluation.

clear; clc; close all;

%% ========================= Project path and configuration =========================
% This block follows the path handling style used in run_03_bayesian_logistic_progression.m.
% It allows this script to be placed inside scripts/... and still find configs/src.
try
    projectRoot = fileparts(mfilename("fullpath"));
    if exist(fullfile(projectRoot, "configs"), "dir")
        addpath(genpath(fullfile(projectRoot, "configs")));
    end
    if exist(fullfile(projectRoot, "src"), "dir")
        addpath(genpath(fullfile(projectRoot, "src")));
    end
catch
    projectRoot = pwd;
end

if exist("config_paths_local", "file") ~= 2
    error(["config_paths_local.m not found. Copy configs/config_paths_template.m " + ...
           "to configs/config_paths_local.m and edit local paths."]);
end
paths = config_paths_local();

%% ========================= User settings =========================
cfg = struct();

% Input/output paths are resolved from configs/config_paths_local.m.

% Repeated subject-level CV
cfg.nRepeats = 30;
cfg.kFold    = 5;
cfg.randomSeed = 42;

% Bayesian logistic regression
cfg.priorStd = 0.5;
cfg.posteriorSamples = 300;

% Feature-set modes
cfg.topKList = [5 7 10 15 20];

% Threshold: "default05" or "trainBA"
cfg.thresholdMode = "trainBA";

% Use inverse-frequency class weights for all models that support weights
cfg.useClassWeights = true;

% Candidate models
cfg.modelNames = ["BLR", "Logistic", "LDA", "SVM_RBF", "BaggedTrees"];

% Preprocessing candidates
cfg.preprocList = makePreprocList();

%% ========================= Resolve paths =========================
cfg.featureFile = fullfile(paths.featureOutputDir, "AllData_FileLevelFeatures.xlsx");
cfg.outputDir = fullfile(paths.outputRoot, "03_benchmark_ml_bayesian");
if ~exist(cfg.outputDir, 'dir'); mkdir(cfg.outputDir); end

diary(fullfile(cfg.outputDir, 'Benchmark_Log.txt'));
fprintf("=== Preprocessing x feature count x model comparison ===\n");
fprintf("Feature file: %s\n", cfg.featureFile);
fprintf("Output dir  : %s\n", cfg.outputDir);
rng(cfg.randomSeed);

%% ========================= Load data =========================
T = readtable(cfg.featureFile, 'VariableNamingRule', 'preserve');
fprintf("Loaded table: %d rows x %d columns\n", height(T), width(T));

[y, taskInfo] = buildProgressionLabels(T);
[Xraw, featureNames] = extractEEGFeatureMatrix(T);
subjectID = extractSubjectIDs(T);

valid = ~isnan(y) & ~all(isnan(Xraw), 2);
Xraw = Xraw(valid, :);
y = y(valid);
subjectID = subjectID(valid);

fprintf("Task label source: %s\n", taskInfo.labelSource);
fprintf("Class counts: Non-decline=%d, Decline=%d\n", sum(y==0), sum(y==1));
fprintf("Feature count: %d\n", numel(featureNames));

dataInfo = table;
dataInfo.Item = ["N_total"; "N_non_decline"; "N_decline"; "N_features"; "LabelSource"];
dataInfo.Value = [string(numel(y)); string(sum(y==0)); string(sum(y==1)); ...
                  string(numel(featureNames)); string(taskInfo.labelSource)];
writetable(dataInfo, fullfile(cfg.outputDir, 'Benchmark_DataInfo.xlsx'));

%% ========================= Experiment loop =========================
rawVarNames = {'Repeat','Fold','Preprocessing','FeatureMode','Model','NFeatures','Threshold', ...
    'NTrain','NTest','NTestDecline','Accuracy','BalancedAccuracy','MacroF1','AUC', ...
    'Brier','NLL','Recall_NonDecline','Recall_Decline','Precision_NonDecline', ...
    'Precision_Decline','MeanPredictiveStd','Status','ErrorMessage'};
rawRows = cell(0, numel(rawVarNames));
selectedFeatureRows = struct([]);
rowCnt = 0;
selCnt = 0;

for rep = 1:cfg.nRepeats
    foldID = makeSubjectStratifiedFolds(subjectID, y, cfg.kFold, cfg.randomSeed + rep);
    fprintf("\nRepeat %d/%d\n", rep, cfg.nRepeats);
    
    for fold = 1:cfg.kFold
        testIdx = (foldID == fold);
        trainIdx = ~testIdx;
        Xtr0 = Xraw(trainIdx, :);
        Xte0 = Xraw(testIdx, :);
        ytr = y(trainIdx);
        yte = y(testIdx);
        fprintf("  Fold %d/%d | train=%d test=%d | test decline=%d\n", ...
            fold, cfg.kFold, numel(ytr), numel(yte), sum(yte==1));
        
        for pp = 1:numel(cfg.preprocList)
            ppCfg = cfg.preprocList(pp);
            [Xtr, prepModel] = fitApplyPreprocess(Xtr0, featureNames, ppCfg);
            Xte = applyPreprocess(Xte0, prepModel);
            featureModes = buildFeatureModes(cfg, featureNames);
            
            for fm = 1:numel(featureModes)
                fmCfg = featureModes(fm);
                
                if fmCfg.Type == "All"
                    selectedIdx = 1:numel(featureNames);
                elseif fmCfg.Type == "TopK"
                    rankScore = rankFeaturesByTrainingOnly(Xtr, ytr);
                    [~, order] = sort(rankScore, 'descend', 'MissingPlacement', 'last');
                    kSel = min(fmCfg.K, numel(featureNames));
                    selectedIdx = sort(order(1:kSel));
                elseif fmCfg.Type == "Fixed"
                    [isMember, selectedIdx] = ismember(fmCfg.FeatureNames, string(featureNames));
                    selectedIdx = selectedIdx(isMember);
                    if isempty(selectedIdx); continue; end
                    selectedIdx = unique(selectedIdx, 'stable');
                else
                    error("Unknown feature mode: %s", fmCfg.Type);
                end
                selectedNames = string(featureNames(selectedIdx));
                
                selCnt = selCnt + 1;
                selectedFeatureRows(selCnt).Repeat = rep;
                selectedFeatureRows(selCnt).Fold = fold;
                selectedFeatureRows(selCnt).Preprocessing = string(ppCfg.Name);
                selectedFeatureRows(selCnt).FeatureMode = string(fmCfg.Name);
                selectedFeatureRows(selCnt).NFeatures = numel(selectedIdx);
                selectedFeatureRows(selCnt).SelectedFeatures = strjoin(selectedNames, "; ");
                
                XtrS = Xtr(:, selectedIdx);
                XteS = Xte(:, selectedIdx);
                sampleWeights = makeClassWeights(ytr, cfg.useClassWeights);
                
                for mm = 1:numel(cfg.modelNames)
                    modelName = cfg.modelNames(mm);
                    try
                        [pTr, pTe, uTe] = fitPredictModel(modelName, XtrS, ytr, XteS, sampleWeights, cfg);
                        thr = chooseThreshold(ytr, pTr, cfg.thresholdMode);
                        metrics = computeMetrics(yte, pTe, thr, uTe);
                        rowCnt = rowCnt + 1;
                        rawRows(rowCnt,:) = makeResultRowCell(rep, fold, ppCfg.Name, fmCfg.Name, modelName, ...
                            numel(selectedIdx), thr, numel(ytr), numel(yte), sum(yte==1), metrics, "OK", "");
                    catch ME
                        rowCnt = rowCnt + 1;
                        rawRows(rowCnt,:) = makeFailedRowCell(rep, fold, ppCfg.Name, fmCfg.Name, modelName, ...
                            numel(selectedIdx), numel(ytr), numel(yte), sum(yte==1), ME.message);
                        fprintf("    FAILED | %s | %s | %s | %s\n", ppCfg.Name, fmCfg.Name, modelName, ME.message);
                    end
                end
            end
        end
    end
    RawPartial = cell2table(rawRows, 'VariableNames', rawVarNames);
    writetable(RawPartial, fullfile(cfg.outputDir, 'Benchmark_RawMetrics_PARTIAL.xlsx'));
end

Raw = cell2table(rawRows, 'VariableNames', rawVarNames);
Selected = struct2table(selectedFeatureRows);
writetable(Raw, fullfile(cfg.outputDir, 'Benchmark_RawMetrics.xlsx'));
writetable(Selected, fullfile(cfg.outputDir, 'Benchmark_SelectedFeatures_ByFold.xlsx'));

%% ========================= Summary tables =========================
metrics = ["Accuracy", "BalancedAccuracy", "MacroF1", "AUC", "Brier", "NLL", ...
           "Recall_NonDecline", "Recall_Decline", "Precision_NonDecline", ...
           "Precision_Decline", "MeanPredictiveStd"];
Summary = summarizeByGroups(Raw, ["Preprocessing", "FeatureMode", "Model", "NFeatures"], metrics);
writetable(Summary, fullfile(cfg.outputDir, 'Benchmark_Summary.xlsx'));
BestBA = sortrows(Summary, "mean_BalancedAccuracy", "descend");
writetable(BestBA, fullfile(cfg.outputDir, 'Benchmark_BestByBalancedAccuracy.xlsx'));
BestAUC = sortrows(Summary, "mean_AUC", "descend");
writetable(BestAUC, fullfile(cfg.outputDir, 'Benchmark_BestByAUC.xlsx'));

%% ========================= Simple figures =========================
try
    figDir = fullfile(cfg.outputDir, 'Figures');
    if ~exist(figDir, 'dir'); mkdir(figDir); end
    Top = BestBA(1:min(20,height(BestBA)), :);
    labels = Top.Preprocessing + " | " + Top.FeatureMode + " | " + Top.Model;
    f = figure('Color','w','Position',[100 100 1200 650]);
    bar(Top.mean_BalancedAccuracy); hold on;
    er = errorbar(1:height(Top), Top.mean_BalancedAccuracy, Top.std_BalancedAccuracy, '.');
    er.LineWidth = 1;
    ylabel('Balanced Accuracy');
    title('Top configurations by repeated-CV balanced accuracy');
    xticks(1:height(Top)); xticklabels(labels); xtickangle(45); grid on;
    saveas(f, fullfile(figDir, 'Fig01_TopBalancedAccuracy.png'));
    savefig(f, fullfile(figDir, 'Fig01_TopBalancedAccuracy.fig'));
    
    BlrSummary = sortrows(Summary(Summary.Model=="BLR",:), "mean_BalancedAccuracy", "descend");
    if ~isempty(BlrSummary)
        bestPP = BlrSummary.Preprocessing(1);
        S2 = Summary(Summary.Model=="BLR" & Summary.Preprocessing==bestPP, :);
        S2 = sortrows(S2, "NFeatures");
        f = figure('Color','w','Position',[100 100 900 550]);
        plot(S2.NFeatures, S2.mean_BalancedAccuracy, '-o', 'LineWidth', 1.5); hold on;
        plot(S2.NFeatures, S2.mean_AUC, '-s', 'LineWidth', 1.5);
        xlabel('Number of selected features'); ylabel('Score');
        title("BLR performance across feature modes | preprocessing = " + bestPP);
        legend({'Balanced Accuracy','AUC'}, 'Location','best'); grid on;
        saveas(f, fullfile(figDir, 'Fig02_BLR_FeatureCount.png'));
        savefig(f, fullfile(figDir, 'Fig02_BLR_FeatureCount.fig'));
    end
catch ME
    warning("Figure generation failed: %s", ME.message);
end

fprintf("\nDone. Results saved to:\n%s\n", cfg.outputDir);
diary off;

%% =================================================================
%% Local functions
%% =================================================================

function preprocList = makePreprocList()
    % Only configurations reported in the manuscript/supplement are exposed.
    preprocList = struct([]);
    names = ["ZScore", "Winsor_ZScore", "SignedLog_ZScore", "Winsor_SignedLog_ZScore"];
    wins = [false true false true];
    slog = [false false true true];
    for i = 1:numel(names)
        preprocList(i).Name = names(i);
        preprocList(i).AbsDiff = false;
        preprocList(i).Winsor = wins(i);
        preprocList(i).SignedLog = slog(i);
        preprocList(i).RobustZ = false;
        preprocList(i).WinsorPct = [1 99];
    end
end

function row = makeResultRowCell(rep, fold, preproc, fmode, model, nfeat, thr, ntr, nte, nDecl, m, status, err)
    % Return one fixed-width cell row instead of a scalar struct.
    % This avoids MATLAB's "subscripted assignment between dissimilar structures" error
    % when success/failure rows or string/char fields are appended repeatedly.
    row = {rep, fold, string(preproc), string(fmode), string(model), nfeat, thr, ...
           ntr, nte, nDecl, m.Accuracy, m.BalancedAccuracy, m.MacroF1, m.AUC, ...
           m.Brier, m.NLL, m.Recall0, m.Recall1, m.Precision0, m.Precision1, ...
           m.MeanPredictiveStd, string(status), string(err)};
end

function row = makeFailedRowCell(rep, fold, preproc, fmode, model, nfeat, ntr, nte, nDecl, err)
    z = nanMetrics();
    row = makeResultRowCell(rep, fold, preproc, fmode, model, nfeat, NaN, ntr, nte, nDecl, z, "FAILED", err);
end

function m = nanMetrics()
    m.Accuracy=NaN; m.BalancedAccuracy=NaN; m.MacroF1=NaN; m.AUC=NaN; m.Brier=NaN; m.NLL=NaN;
    m.Recall0=NaN; m.Recall1=NaN; m.Precision0=NaN; m.Precision1=NaN; m.MeanPredictiveStd=NaN;
end

function modes = buildFeatureModes(cfg, featureNames)
    modes = struct([]); c = 1;
    modes(c).Name = "All"; modes(c).Type = "All"; modes(c).K = numel(featureNames); modes(c).FeatureNames = string.empty;
    for ii = 1:numel(cfg.topKList)
        c = c + 1;
        modes(c).Name = "TopK_" + string(cfg.topKList(ii)); modes(c).Type = "TopK";
        modes(c).K = cfg.topKList(ii); modes(c).FeatureNames = string.empty;
    end
end

function [y, info] = buildProgressionLabels(T)
    % Build the exact target label used in run_03_bayesian_logistic_progression.m
    % Target task:
    %   Class 0: Maintain_CN + Maintain_SCD  -> Maintain_CN_SCD
    %   Class 1: Decline_MCI_to_DEM + Decline_Normal_to_DEM -> Decline_to_DEM
    %
    % Important: the raw Strict_Subtype column is not always already in this simplified form.
    % Therefore, this function first reconstructs AnalysisSubtype using Strict_Label,
    % Strict_Subtype, and Current_Diagnosis, matching run_03.

    vars = string(T.Properties.VariableNames);
    info = struct();
    info.labelSource = "AnalysisSubtype_from_StrictLabel_StrictSubtype_CurrentDiagnosis";
    info.ClassAName = "Maintain_CN_SCD";
    info.ClassBName = "Decline_to_DEM";
    info.ClassAValues = ["Maintain_CN", "Maintain_SCD"];
    info.ClassBValues = ["Decline_MCI_to_DEM", "Decline_Normal_to_DEM"];

    y = nan(height(T), 1);

    % 1) If an AnalysisSubtype column already exists, use it directly.
    if ismember("AnalysisSubtype", vars)
        analysisSubtype = string(strtrim(T.AnalysisSubtype));
        info.labelSource = "AnalysisSubtype";
    else
        % 2) Otherwise, reproduce run_03 logic.
        requiredVars = ["Strict_Label", "Strict_Subtype", "Current_Diagnosis"];
        for ii = 1:numel(requiredVars)
            if ~ismember(requiredVars(ii), vars)
                error("%s column is required to build progression labels.", requiredVars(ii));
            end
        end

        strictLabel = string(strtrim(T.Strict_Label));
        strictSubtype = string(strtrim(T.Strict_Subtype));
        currentDx = normalizeDx(T.Current_Diagnosis);

        analysisSubtype = simplifyProgressionSubtype(strictLabel, strictSubtype, currentDx);
    end

    idxA = ismember(analysisSubtype, info.ClassAValues);
    idxB = ismember(analysisSubtype, info.ClassBValues);

    y(idxA) = 0;
    y(idxB) = 1;

    fprintf("\nLabel construction check\n");
    fprintf("  Source: %s\n", info.labelSource);
    fprintf("  Maintain_CN     : %d\n", sum(analysisSubtype == "Maintain_CN"));
    fprintf("  Maintain_SCD    : %d\n", sum(analysisSubtype == "Maintain_SCD"));
    fprintf("  Decline_MCI_DEM : %d\n", sum(analysisSubtype == "Decline_MCI_to_DEM"));
    fprintf("  Decline_Norm_DEM: %d\n", sum(analysisSubtype == "Decline_Normal_to_DEM"));
    fprintf("  Used Class 0    : %d\n", sum(y == 0));
    fprintf("  Used Class 1    : %d\n", sum(y == 1));
    fprintf("  Excluded/NaN    : %d\n", sum(isnan(y))); 

    if sum(y == 0) == 0 || sum(y == 1) == 0
        warning("One of the target classes has zero samples. Writing AnalysisSubtype counts for debugging.");
        disp(countByStringLocal(analysisSubtype, "AnalysisSubtype"));
        error("Target label construction failed: Class 0 or Class 1 is empty.");
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

function summaryTable = countByStringLocal(x, varName)
    x = string(x);
    [G, names] = findgroups(x);
    counts = splitapply(@numel, x, G);
    summaryTable = table(names, counts, 'VariableNames', {char(varName), 'Count'});
end

function [X, featureNames] = extractEEGFeatureMatrix(T)
    % Match the feature-loading logic used in run_03: use all numeric columns
    % except metadata/progression columns. This avoids losing valid EEG features
    % just because their names do not match a narrow keyword filter.

    metaVars = [
        "FileKey"
        "Strict_Label"
        "Strict_Subtype"
        "AnalysisSubtype"
        "Current_Diagnosis"
        "ObjectID"
        "NumSegments"
        "FullPath"
        "BinaryLabel"
        "TaskLabel"
        "Label"
        "Group"
        "Class"
    ];

    allVars = string(T.Properties.VariableNames);
    featureVars = setdiff(allVars, metaVars, "stable");

    isNum = false(size(featureVars));
    for i = 1:numel(featureVars)
        isNum(i) = isnumeric(T.(featureVars(i)));
    end

    featureVars = featureVars(isNum);

    if isempty(featureVars)
        error("No numeric feature columns were detected after excluding metadata variables.");
    end

    X = table2array(T(:, cellstr(featureVars)));
    X = double(X);

    % Missing / invalid value handling, matching run_03 style.
    for j = 1:size(X, 2)
        col = X(:, j);
        medVal = median(col, "omitnan");
        if isnan(medVal)
            medVal = 0;
        end
        col(~isfinite(col)) = medVal;
        X(:, j) = col;
    end

    featureNames = cellstr(featureVars);
end

function subjectID = extractSubjectIDs(T)
    vars = string(T.Properties.VariableNames); candidates = ["ObjectID","SubjectID","subject_id","ID","id","OID","oid"];
    idx = [];
    for c = candidates
        idx = find(strcmpi(vars, c), 1); if ~isempty(idx); break; end
    end
    if isempty(idx)
        warning("No subject ID column found. Each row is treated as an independent subject."); subjectID = string((1:height(T))');
    else
        subjectID = string(T{:,idx});
    end
end

function foldID = makeSubjectStratifiedFolds(subjectID, y, K, seed)
    rng(seed); [uSub, ~, subIdx] = unique(subjectID, 'stable'); nSub = numel(uSub); subY = zeros(nSub,1);
    for s = 1:nSub; subY(s) = max(y(subIdx==s)); end
    subFold = zeros(nSub,1);
    for cls = [0 1]
        ids = find(subY == cls); ids = ids(randperm(numel(ids)));
        for ii = 1:numel(ids); subFold(ids(ii)) = mod(ii-1, K) + 1; end
    end
    foldID = subFold(subIdx);
end

function [XtrP, model] = fitApplyPreprocess(Xtr, featureNames, ppCfg)
    Xtr = double(Xtr); Xtr(~isfinite(Xtr)) = NaN; model = struct(); model.featureNames = string(featureNames); model.ppCfg = ppCfg;
    med = median(Xtr, 1, 'omitnan'); med(isnan(med)) = 0; model.imputeMedian = med; Xtr = fillmissingWithVector(Xtr, med);
    model.isDiff = detectDiffFeatures(featureNames); if ppCfg.AbsDiff; Xtr(:, model.isDiff) = abs(Xtr(:, model.isDiff)); end
    if ppCfg.Winsor; lo = prctile(Xtr, ppCfg.WinsorPct(1), 1); hi = prctile(Xtr, ppCfg.WinsorPct(2), 1); else; lo = -inf(1,size(Xtr,2)); hi = inf(1,size(Xtr,2)); end
    model.lo = lo; model.hi = hi; Xtr = min(max(Xtr, lo), hi);
    if ppCfg.SignedLog; Xtr = sign(Xtr) .* log1p(abs(Xtr)); end
    if ppCfg.RobustZ
        center = median(Xtr, 1, 'omitnan'); scale = iqr(Xtr, 1); scale(scale < 1e-8 | isnan(scale)) = 1;
    else
        center = mean(Xtr, 1, 'omitnan'); scale = std(Xtr, 0, 1, 'omitnan'); scale(scale < 1e-8 | isnan(scale)) = 1;
    end
    model.center = center; model.scale = scale; XtrP = (Xtr - center) ./ scale; XtrP(~isfinite(XtrP)) = 0;
end

function XteP = applyPreprocess(Xte, model)
    Xte = double(Xte); Xte(~isfinite(Xte)) = NaN; Xte = fillmissingWithVector(Xte, model.imputeMedian);
    if model.ppCfg.AbsDiff; Xte(:, model.isDiff) = abs(Xte(:, model.isDiff)); end
    Xte = min(max(Xte, model.lo), model.hi);
    if model.ppCfg.SignedLog; Xte = sign(Xte) .* log1p(abs(Xte)); end
    XteP = (Xte - model.center) ./ model.scale; XteP(~isfinite(XteP)) = 0;
end

function X = fillmissingWithVector(X, v)
    for j = 1:size(X,2); idx = isnan(X(:,j)); if any(idx); X(idx,j) = v(j); end; end
end

function isDiff = detectDiffFeatures(featureNames)
    s = lower(string(featureNames));
    isDiff = contains(s,"diff") | contains(s,"delta") | contains(s,"chdiff") | contains(s,"ch1_ch2") | contains(s,"ch2_ch1") | contains(s,"_d_") | startsWith(s,"d_");
end

function score = rankFeaturesByTrainingOnly(X, y)
    score = zeros(1, size(X,2));
    for j = 1:size(X,2)
        x0 = X(y==0,j); x1 = X(y==1,j);
        m0 = mean(x0, 'omitnan'); m1 = mean(x1, 'omitnan'); s0 = var(x0, 'omitnan'); s1 = var(x1, 'omitnan');
        score(j) = abs(m1-m0) / sqrt(0.5*(s0+s1) + 1e-8); if ~isfinite(score(j)); score(j)=0; end
    end
end

function w = makeClassWeights(y, useWeights)
    w = ones(size(y)); if ~useWeights; return; end
    n = numel(y); n0 = sum(y==0); n1 = sum(y==1);
    if n0 > 0; w(y==0) = n/(2*n0); end
    if n1 > 0; w(y==1) = n/(2*n1); end
end

function [pTr, pTe, uTe] = fitPredictModel(modelName, Xtr, ytr, Xte, sampleWeights, cfg)
    modelName = string(modelName);
    switch modelName
        case "BLR"
            mdl = fitBLR_Laplace(Xtr, ytr, sampleWeights, cfg.priorStd);
            [pTr, ~] = predictBLR_Laplace(mdl, Xtr, cfg.posteriorSamples);
            [pTe, uTe] = predictBLR_Laplace(mdl, Xte, cfg.posteriorSamples);
        case "Logistic"
            mdl = fitclinear(Xtr, ytr, 'Learner','logistic','Regularization','ridge','Lambda',1e-4,'Weights',sampleWeights,'ClassNames',[0;1]);
            [~, sTr] = predict(mdl, Xtr); [~, sTe] = predict(mdl, Xte); pTr = scoreToProb(sTr, mdl.ClassNames); pTe = scoreToProb(sTe, mdl.ClassNames); uTe = nan(size(pTe));
        case "LDA"
            mdl = fitcdiscr(Xtr, ytr, 'DiscrimType','pseudoLinear','Weights',sampleWeights,'ClassNames',[0;1]);
            [~, sTr] = predict(mdl, Xtr); [~, sTe] = predict(mdl, Xte); pTr = scoreToProb(sTr, mdl.ClassNames); pTe = scoreToProb(sTe, mdl.ClassNames); uTe = nan(size(pTe));
        case "SVM_RBF"
            mdl0 = fitcsvm(Xtr, ytr, 'KernelFunction','rbf','KernelScale','auto','Standardize',false,'Weights',sampleWeights,'ClassNames',[0;1]);
            try
                mdl = fitPosterior(mdl0, Xtr, ytr); [~, sTr] = predict(mdl, Xtr); [~, sTe] = predict(mdl, Xte); pTr = scoreToProb(sTr, mdl.ClassNames); pTe = scoreToProb(sTe, mdl.ClassNames);
            catch
                [~, sTr] = predict(mdl0, Xtr); [~, sTe] = predict(mdl0, Xte); col = find(mdl0.ClassNames == 1, 1); pTr = safeSigmoid(sTr(:,col)); pTe = safeSigmoid(sTe(:,col));
            end
            uTe = nan(size(pTe));
        case "BaggedTrees"
            t = templateTree('MaxNumSplits', max(2, min(20, floor(size(Xtr,1)/5))));
            mdl = fitcensemble(Xtr, ytr, 'Method','Bag','NumLearningCycles',100,'Learners',t,'Weights',sampleWeights,'ClassNames',[0;1]);
            [~, sTr] = predict(mdl, Xtr); [~, sTe] = predict(mdl, Xte); pTr = scoreToProb(sTr, mdl.ClassNames); pTe = scoreToProb(sTe, mdl.ClassNames); uTe = nan(size(pTe));
        otherwise
            error("Unknown model: %s", modelName);
    end
    pTr = min(max(pTr(:), 1e-6), 1-1e-6); pTe = min(max(pTe(:), 1e-6), 1-1e-6);
end

function p = scoreToProb(score, classNames)
    col = find(classNames == 1, 1); if isempty(col); col = size(score,2); end
    p = score(:, col); if any(p < 0) || any(p > 1); p = safeSigmoid(p); end
end

function mdl = fitBLR_Laplace(X, y, w, priorStd)
    Xb = [ones(size(X,1),1), X]; p = size(Xb,2); b0 = zeros(p,1); obj = @(b) blrObjectiveGradient(b, Xb, y, w, priorStd);
    if exist('fminunc','file') == 2
        opts = optimoptions('fminunc','Display','off','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIterations',300,'OptimalityTolerance',1e-6);
        beta = fminunc(obj, b0, opts);
    else
        beta = fminsearch(@(bb) blrObjectiveOnly(bb, Xb, y, w, priorStd), b0);
    end
    prob = safeSigmoid(Xb * beta); W = w(:).*prob.*(1-prob); priorPrec = diag([0; ones(p-1,1)] ./ (priorStd^2));
    H = Xb' * (Xb .* W) + priorPrec; H = (H + H')/2;
    jitter = 1e-6; ok = false;
    for t = 1:8; [R, flag] = chol(H + jitter*eye(p)); if flag == 0; ok = true; break; end; jitter = jitter*10; end
    if ok; covBeta = inv(R) * inv(R)'; else; covBeta = pinv(H + jitter*eye(p)); end
    mdl.beta = beta; mdl.covBeta = (covBeta + covBeta')/2;
end

function [f, g] = blrObjectiveGradient(beta, Xb, y, w, priorStd)
    eta = Xb * beta; p = safeSigmoid(eta); p = min(max(p,1e-8),1-1e-8);
    nll = -sum(w(:) .* (y(:).*log(p) + (1-y(:)).*log(1-p))); reg = 0.5 * sum(beta(2:end).^2)/(priorStd^2); f = nll + reg;
    g = Xb' * (w(:).*(p-y(:))); g(2:end) = g(2:end) + beta(2:end)/(priorStd^2);
end

function f = blrObjectiveOnly(beta, Xb, y, w, priorStd)
    f = blrObjectiveGradient(beta, Xb, y, w, priorStd);
end

function [pMean, pStd] = predictBLR_Laplace(mdl, X, S)
    Xb = [ones(size(X,1),1), X]; pDim = numel(mdl.beta); covB = (mdl.covBeta + mdl.covBeta')/2;
    jitter = 1e-8; ok = false;
    for t = 1:8; [R, flag] = chol(covB + jitter*eye(pDim)); if flag == 0; ok = true; break; end; jitter = jitter*10; end
    if ok; betaSamples = randn(S, pDim)*R + mdl.beta(:)'; else; sd = sqrt(max(diag(covB),0))'; betaSamples = randn(S,pDim).*sd + mdl.beta(:)'; end
    prob = safeSigmoid(Xb * betaSamples'); pMean = mean(prob,2); pStd = std(prob,0,2);
end

function p = safeSigmoid(z)
    z = max(min(z, 40), -40); p = 1 ./ (1 + exp(-z));
end

function thr = chooseThreshold(y, p, mode)
    mode = string(mode);
    if mode == "default05"; thr = 0.5; return; end
    if mode == "trainBA"
        cand = unique([0.05; 0.1; (0.15:0.01:0.85)'; 0.9; 0.95; p(:)]);
        bestBA = -inf; bestThr = 0.5;
        for ii = 1:numel(cand)
            m = computeMetrics(y, p, cand(ii), nan(size(p)));
            if m.BalancedAccuracy > bestBA; bestBA = m.BalancedAccuracy; bestThr = cand(ii); end
        end
        thr = bestThr; return;
    end
    error("Unknown threshold mode: %s", mode);
end

function m = computeMetrics(y, p, thr, u)
    y = y(:); p = p(:); pred = double(p >= thr);
    TP1 = sum(pred==1 & y==1); TN0 = sum(pred==0 & y==0); FP1 = sum(pred==1 & y==0); FN1 = sum(pred==0 & y==1);
    recall1 = safeDiv(TP1, TP1+FN1); recall0 = safeDiv(TN0, TN0+FP1); prec1 = safeDiv(TP1, TP1+FP1); prec0 = safeDiv(TN0, TN0+FN1);
    f1_1 = safeDiv(2*prec1*recall1, prec1+recall1); f1_0 = safeDiv(2*prec0*recall0, prec0+recall0);
    m = struct(); m.Accuracy = mean(pred == y); m.BalancedAccuracy = mean([recall0 recall1], 'omitnan'); m.Recall0 = recall0; m.Recall1 = recall1;
    m.Precision0 = prec0; m.Precision1 = prec1; m.MacroF1 = mean([f1_0 f1_1], 'omitnan'); m.Brier = mean((p-y).^2, 'omitnan');
    pClip = min(max(p,1e-6),1-1e-6); m.NLL = -mean(y.*log(pClip) + (1-y).*log(1-pClip), 'omitnan');
    if numel(unique(y)) == 2 && exist('perfcurve','file') == 2
        try; [~,~,~,auc] = perfcurve(y, p, 1); catch; auc = NaN; end
    else; auc = NaN; end
    m.AUC = auc; m.MeanPredictiveStd = mean(u, 'omitnan');
end

function v = safeDiv(a,b)
    if b == 0; v = NaN; else; v = a/b; end
end

function Summary = summarizeByGroups(Raw, groupVars, metrics)
    RawOK = Raw(Raw.Status=="OK", :); [G, groupTable] = findgroups(RawOK(:, cellstr(groupVars))); Summary = groupTable;
    for m = metrics
        x = RawOK.(m); Summary.("mean_"+m) = splitapply(@(v) mean(v,'omitnan'), x, G); Summary.("std_"+m) = splitapply(@(v) std(v,'omitnan'), x, G);
    end
    Summary.NRuns = splitapply(@numel, RawOK.Accuracy, G);
end
