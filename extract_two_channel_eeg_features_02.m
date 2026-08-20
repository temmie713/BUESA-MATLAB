%% 02_extract_two_channel_eeg_features.m
% Preprocess two-channel EEG and extract the 64 file-level features used in
% the BUESA study.
%
% Signal processing:
%   - channels 1 and 2, fs = 250 Hz
%   - linear missing-value interpolation, median centering, detrending
%   - 4th-order zero-phase Butterworth band-pass filtering (1-45 Hz)
%   - robust z-normalization and clipping to [-8, 8]
%   - 10-s segments with 50% overlap; maximum 40 accepted segments/file
%   - artifact rejection using amplitude, variance, derivative, and high-
%     amplitude-ratio criteria
%   - Welch spectral descriptors and two-channel summary descriptors
%   - mean/std/median/IQR aggregation to 64 file-level features

clc; clear; close all;

%% =========================================================
% 0. Paths and settings
%% =========================================================
projectRoot = fileparts(mfilename("fullpath"));
addpath(fullfile(projectRoot, "configs"));
if exist("config_paths_local", "file") ~= 2
    error(["config_paths_local.m not found. Copy configs/config_paths_template.m " + ...
           "to configs/config_paths_local.m and edit local paths."]);
end
paths = config_paths_local();

fs = 250;
segmentSec = 10;
overlapRatio = 0.5;
maxSegmentsPerFile = 40;
channelsToUse = [1 2];

listFile = fullfile(paths.labelOutputDir, "EEG_strict_progression_final_list.xlsx");
beamRoot = paths.beamRoot;
outDir = paths.featureOutputDir;
if ~exist(outDir, "dir"); mkdir(outDir); end

rng(42);

%% =========================================================
% 1. Strict progression list 불러오기
%% =========================================================

T = readtable(listFile, "VariableNamingRule","preserve");

fprintf("\n=== Strict progression list ===\n");
fprintf("row 수: %d\n", height(T));

if ~ismember("Strict_Label", T.Properties.VariableNames)
    error("Strict_Label 컬럼을 찾을 수 없습니다.");
end

if ~ismember("BEAM_File", T.Properties.VariableNames)
    error("BEAM_File 컬럼을 찾을 수 없습니다.");
end

T.Strict_Label = string(strtrim(T.Strict_Label));

% Maintain / Declining만 사용
T = T(T.Strict_Label == "Maintain" | T.Strict_Label == "Declining", :);

fprintf("\n=== Strict_Label 분포 ===\n");
disp(countByString(T.Strict_Label, "Strict_Label"));

%% =========================================================
% 2. BEAM txt 파일 전체 검색 및 매칭
%% =========================================================

allTxt = dir(fullfile(beamRoot, "**", "*.txt"));

fprintf("\nBEAM 폴더 내부 txt 파일 수: %d개\n", numel(allTxt));

fileTable = table();

for i = 1:numel(allTxt)

    fullPath = fullfile(allTxt(i).folder, allTxt(i).name);
    baseName = erase(string(allTxt(i).name), ".txt");

    newRow = table( ...
        string(allTxt(i).name), ...
        baseName, ...
        string(fullPath), ...
        'VariableNames', ...
        ["FileName", "BaseName", "FullPath"]);

    fileTable = [fileTable; newRow];
end

T.BEAM_Key = normalizeEEGFileName(T.BEAM_File);
fileTable.BEAM_Key = normalizeEEGFileName(fileTable.BaseName);

[isFound, loc] = ismember(T.BEAM_Key, fileTable.BEAM_Key);

T.FileFound = isFound;
T.FullPath = strings(height(T),1);
T.FullPath(isFound) = fileTable.FullPath(loc(isFound));

fprintf("\nStrict list BEAM 파일 매칭 성공: %d / %d\n", ...
    sum(isFound), height(T));

unmatched = T(~isFound, :);
if ~isempty(unmatched)
    writetable(unmatched, fullfile(outDir, "BEAM_Unmatched_StrictList.xlsx"));
end

T = T(isFound, :);

fprintf("\n=== 매칭 후 Strict_Label 분포 ===\n");
disp(countByString(T.Strict_Label, "Strict_Label"));

%% =========================================================
% 3. Extract features from all matched eligible EEG records
%% =========================================================
% The manuscript analysis retains all eligible records. No majority-class
% downsampling is performed at this stage.
T_all = T;
writetable(T_all, fullfile(outDir, "StrictProgression_BEAM_Matched_All.xlsx"));

fprintf("\n\n============================================\n");
fprintf("Feature extraction: all matched records\n");
fprintf("============================================\n");

runOneDatasetAnalysis(T_all, "AllData", outDir, fs, segmentSec, overlapRatio, ...
    maxSegmentsPerFile, channelsToUse);

fprintf("\nFeature extraction complete. Output folder: %s\n", outDir);

%% =========================================================
%% Main analysis function
%% =========================================================

function runOneDatasetAnalysis(T_use, tag, outDir, fs, segmentSec, overlapRatio, maxSegmentsPerFile, channelsToUse)

    %% -----------------------------------------------------
    % 1. Segment feature 추출
    %% -----------------------------------------------------
    segmentFeatureTable = table();

    for i = 1:height(T_use)

        filePath = T_use.FullPath(i);
        label = T_use.Strict_Label(i);
        fileKey = T_use.BEAM_Key(i);

        if ismember("ObjectID", T_use.Properties.VariableNames)
            obj = T_use.ObjectID(i);
        else
            obj = i;
        end

        try
            [X, ~] = readEEGTxt(filePath);

            if size(X,2) < max(channelsToUse)
                fprintf("채널 부족: %s\n", filePath);
                continue;
            end

            X = X(:, channelsToUse);

            % strict preprocessing
            X = preprocessEEG_strict(X, fs);

            % segment 분할
            segList = makeSegments(X, fs, segmentSec, overlapRatio);

            if isempty(segList)
                fprintf("segment 없음: %s\n", filePath);
                continue;
            end

            % 파일당 최대 segment 제한
            nSeg = numel(segList);

            if nSeg > maxSegmentsPerFile
                sel = randperm(nSeg, maxSegmentsPerFile);
                segList = segList(sel);
                nSeg = maxSegmentsPerFile;
            end

            for s = 1:nSeg

                segX = segList{s};

                feat = extractBEAMFeatures(segX, fs, channelsToUse);

                row = array2table(feat.values, ...
                    'VariableNames', feat.names);

                row.Strict_Label = label;
                row.FileKey = fileKey;
                row.FullPath = filePath;
                row.ObjectID = obj;
                row.SegmentID = s;

                if ismember("Strict_Subtype", T_use.Properties.VariableNames)
                    row.Strict_Subtype = string(T_use.Strict_Subtype(i));
                end

                if ismember("Current_Diagnosis", T_use.Properties.VariableNames)
                    row.Current_Diagnosis = string(T_use.Current_Diagnosis(i));
                end

                segmentFeatureTable = [segmentFeatureTable; row];
            end

        catch ME
            fprintf("파일 처리 실패: %s\n", filePath);
            fprintf("이유: %s\n", ME.message);
        end
    end

    fprintf("\n[%s] Segment feature 추출 완료: %d segments\n", tag, height(segmentFeatureTable));
    fprintf("[%s] 사용된 파일 수: %d files\n", tag, numel(unique(segmentFeatureTable.FileKey)));
    disp(countByString(segmentFeatureTable.Strict_Label, "Strict_Label"));

    writetable(segmentFeatureTable, ...
        fullfile(outDir, tag + "_SegmentFeatures.xlsx"));

    %% -----------------------------------------------------
    % 2. File-level aggregation
    %% -----------------------------------------------------
    FILEDATA = aggregateSegmentToFile(segmentFeatureTable);

    fprintf("\n[%s] File-level feature table\n", tag);
    fprintf("파일 수: %d\n", height(FILEDATA));
    disp(countByString(FILEDATA.Strict_Label, "Strict_Label"));

    writetable(FILEDATA, ...
        fullfile(outDir, tag + "_FileLevelFeatures.xlsx"));

    fprintf("[%s] Saved %d file-level rows.\n", tag, height(FILEDATA));
end


%% =========================================================
%% Helper functions
%% =========================================================

function [X, t] = readEEGTxt(filePath)

    filePath = string(filePath);

    fid = fopen(filePath, 'r');

    if fid == -1
        error("파일을 열 수 없습니다: %s", filePath);
    end

    numericStartLine = 0;
    lineCount = 0;

    while ~feof(fid)
        line = fgetl(fid);
        lineCount = lineCount + 1;

        if ~ischar(line)
            continue;
        end

        lineStr = strtrim(string(line));

        if ~isempty(regexp(lineStr, "^[+-]?\d+(\.\d+)?", "once"))
            numericStartLine = lineCount;
            break;
        end
    end

    fclose(fid);

    if numericStartLine == 0
        error("숫자 데이터 시작 줄을 찾지 못했습니다.");
    end

    opts = detectImportOptions(filePath, "FileType","text");
    opts.DataLines = [numericStartLine Inf];

    data = readmatrix(filePath, opts);

    if size(data,2) < 3
        error("데이터 열 수 부족.");
    end

    t = data(:,1);
    X = data(:,2:end);

    if all(isnan(t)) || numel(unique(t)) < 2
        fs = 250;
        t = (0:size(X,1)-1)' / fs;
    end
end


function Xout = preprocessEEG_strict(X, fs)

    X = double(X);

    for ch = 1:size(X,2)

        x = X(:,ch);

        if any(isnan(x))
            x = fillmissing(x, "linear", "EndValues","nearest");
        end

        x = x - median(x, "omitnan");
        x = detrend(x);

        % bandpass 1-45 Hz
        [b,a] = butter(4, [1 45]/(fs/2), "bandpass");
        x = filtfilt(b,a,x);

        % robust z-score
        medx = median(x, "omitnan");
        madx = mad(x, 1);

        if madx < eps
            madx = std(x, "omitnan") + eps;
        end

        x = (x - medx) / (1.4826 * madx + eps);

        % winsorize
        x(x > 8) = 8;
        x(x < -8) = -8;

        X(:,ch) = x;
    end

    Xout = X;
end


function segList = makeSegments(X, fs, segmentSec, overlapRatio)

    segmentLen = round(fs * segmentSec);
    hopLen = round(segmentLen * (1 - overlapRatio));

    if hopLen < 1
        hopLen = segmentLen;
    end

    n = size(X,1);
    segList = {};

    if n < segmentLen
        return;
    end

    for st = 1:hopLen:(n - segmentLen + 1)

        ed = st + segmentLen - 1;
        segX = X(st:ed, :);

        if isBadEEGSegment(segX)
            continue;
        end

        segList{end+1,1} = segX;
    end
end


function isBad = isBadEEGSegment(segX)

    isBad = false;

    if any(~isfinite(segX), "all")
        isBad = true;
        return;
    end

    if max(abs(segX), [], "all") > 8
        isBad = true;
        return;
    end

    if any(std(segX,0,1) < 0.05)
        isBad = true;
        return;
    end

    dx = diff(segX);

    if max(abs(dx), [], "all") > 5
        isBad = true;
        return;
    end

    highAmpRatio = mean(abs(segX) > 5, "all");

    if highAmpRatio > 0.05
        isBad = true;
        return;
    end
end


function feat = extractBEAMFeatures(X, fs, channelsToUse)

    X = double(X);
    nCh = size(X,2);

    names = {};
    values = [];

    thetaAlpha = nan(1,nCh);
    medianFreqs = nan(1,nCh);
    sef95s = nan(1,nCh);

    for ch = 1:nCh

        x = X(:,ch);
        originalCh = channelsToUse(ch);

        winLen = min(length(x), fs*2);
        noverlap = floor(winLen/2);
        fgrid = 0:0.5:45;

        [pxx, f] = pwelch(x, hamming(winLen), noverlap, fgrid, fs);

        totalPower = bandpower(pxx, f, [1 45], "psd");
        thetaPower = bandpower(pxx, f, [4 8], "psd");
        alphaPower = bandpower(pxx, f, [8 13], "psd");
        betaPower  = bandpower(pxx, f, [13 30], "psd");

        thetaRel = thetaPower / (totalPower + eps);
        alphaRel = alphaPower / (totalPower + eps);
        betaRel  = betaPower  / (totalPower + eps);

        thetaAlpha(ch) = thetaPower / (alphaPower + eps);

        csum = cumsum(pxx);
        idxMed = find(csum >= csum(end)*0.5, 1);
        idxSEF = find(csum >= csum(end)*0.95, 1);

        medianFreqs(ch) = f(idxMed);
        sef95s(ch) = f(idxSEF);

        names{end+1} = sprintf("ch%d_theta_rel", originalCh);
        values(end+1) = thetaRel;

        names{end+1} = sprintf("ch%d_alpha_rel", originalCh);
        values(end+1) = alphaRel;

        names{end+1} = sprintf("ch%d_beta_rel", originalCh);
        values(end+1) = betaRel;

        names{end+1} = sprintf("ch%d_theta_alpha_ratio", originalCh);
        values(end+1) = thetaAlpha(ch);

        names{end+1} = sprintf("ch%d_median_freq", originalCh);
        values(end+1) = medianFreqs(ch);

        names{end+1} = sprintf("ch%d_sef95", originalCh);
        values(end+1) = sef95s(ch);
    end

    if nCh >= 2
        for a = 1:nCh-1
            for b = a+1:nCh
                chA = channelsToUse(a);
                chB = channelsToUse(b);

                c = corr(X(:,a), X(:,b), "Rows","complete");

                names{end+1} = sprintf("ch%d_ch%d_corr", chA, chB);
                values(end+1) = c;
            end
        end
    end

    names{end+1} = "mean_theta_alpha_ratio";
    values(end+1) = mean(thetaAlpha, "omitnan");

    names{end+1} = "mean_median_freq";
    values(end+1) = mean(medianFreqs, "omitnan");

    names{end+1} = "mean_sef95";
    values(end+1) = mean(sef95s, "omitnan");

    names = cellfun(@char, names, 'UniformOutput', false);

    feat.names = matlab.lang.makeValidName(names);
    feat.values = values(:)';
end


function FILEDATA = aggregateSegmentToFile(segmentFeatureTable)

    fileKeys = unique(segmentFeatureTable.FileKey, "stable");

    metaVars = [
        "Strict_Label"
        "FileKey"
        "FullPath"
        "ObjectID"
        "SegmentID"
        "Strict_Subtype"
        "Current_Diagnosis"
    ];

    allVars = string(segmentFeatureTable.Properties.VariableNames);
    featureVars = setdiff(allVars, metaVars, "stable");

    isNum = false(size(featureVars));
    for i = 1:numel(featureVars)
        isNum(i) = isnumeric(segmentFeatureTable.(featureVars(i)));
    end
    featureVars = featureVars(isNum);

    FILEDATA = table();

    for i = 1:numel(fileKeys)

        fk = fileKeys(i);
        rows = segmentFeatureTable(segmentFeatureTable.FileKey == fk, :);

        newRow = table();
        newRow.FileKey = fk;
        newRow.Strict_Label = rows.Strict_Label(1);
        newRow.ObjectID = rows.ObjectID(1);
        newRow.NumSegments = height(rows);
        newRow.FullPath = rows.FullPath(1);

        if ismember("Strict_Subtype", rows.Properties.VariableNames)
            newRow.Strict_Subtype = rows.Strict_Subtype(1);
        end

        if ismember("Current_Diagnosis", rows.Properties.VariableNames)
            newRow.Current_Diagnosis = rows.Current_Diagnosis(1);
        end

        for j = 1:numel(featureVars)

            f = featureVars(j);
            x = rows.(f);

            newRow.(char(f + "_mean")) = mean(x, "omitnan");
            newRow.(char(f + "_std"))  = std(x, 0, "omitnan");
            newRow.(char(f + "_med"))  = median(x, "omitnan");
            newRow.(char(f + "_iqr"))  = iqr(x);
        end

        FILEDATA = [FILEDATA; newRow];
    end
end


function key = normalizeEEGFileName(name)

    name = string(name);
    key = strings(size(name));

    for i = 1:numel(name)

        s = lower(strtrim(name(i)));
        s = erase(s, ".txt");
        s = regexprep(s, "\s+", "");

        tok = regexp(s, ...
            "id(\d+)-(\d{4})y-(\d{2})m_(\d{2})d_(\d{1,2})h_(\d{1,2})m", ...
            "tokens");

        if ~isempty(tok)
            t = tok{1};

            key(i) = "id" + t{1} + "_" + ...
                     t{2} + t{3} + t{4} + "_" + ...
                     sprintf("%02d%02d", ...
                     str2double(t{5}), str2double(t{6}));
        else
            s = regexprep(s, "(_[a-z]_raw|_raw|-raw)$", "");
            s = regexprep(s, "\([^)]*\)", "");
            s = regexprep(s, "_(\d{1,2})h_(\d{1,2})m_\d{1,2}s", "_$1h_$2m");
            s = regexprep(s, "_\(l\d+_r\d+\)", "");
            s = regexprep(s, "_+$", "");

            key(i) = s;
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