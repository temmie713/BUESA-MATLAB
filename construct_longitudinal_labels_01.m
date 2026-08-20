%% 01_construct_longitudinal_labels.m
% Construct longitudinal EEG progression labels used by the BUESA study.
%
% The script matches each EEG examination to the nearest clinical diagnosis
% within a configurable month tolerance and assigns a strict longitudinal
% trajectory label. Stable CN/SCD requires the same diagnosis across the
% previous/current/next visits. Declining trajectories must contain at least
% one worsening transition and no improving or oscillating transition.
%
% Public-repository note:
%   Raw GARD data are not distributed with this repository. Copy
%   configs/config_paths_template.m to configs/config_paths_local.m and edit
%   local paths before running this script.

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

eegFile  = paths.eegListFile;
diaFile  = paths.diagnosisFile;
snsbFile = paths.snsbFile;  % optional auxiliary export; not required for class assignment

if ~exist(paths.labelOutputDir, "dir"); mkdir(paths.labelOutputDir); end
outAll   = fullfile(paths.labelOutputDir, "EEG_strict_progression_candidate_all.xlsx");
outFinal = fullfile(paths.labelOutputDir, "EEG_strict_progression_final_list.xlsx");

maxMonthDiff = 3;   % maximum EEG-to-diagnosis matching difference in months

%% =========================================================
% 1. 데이터 불러오기
%% =========================================================
EEG  = readTablePreserve(eegFile);
DIA  = readTablePreserve(diaFile);
if isfile(snsbFile)
    SNSB = readTablePreserve(snsbFile);
    hasSNSB = true;
else
    SNSB = table();
    hasSNSB = false;
    warning("Optional SNSB/K-MMSE file not found. Label assignment is unaffected; auxiliary score fields will be NaN.");
end

%% =========================================================
% 2. 컬럼명 설정
%% =========================================================
% brainwave_7175_oid_list
colObjNew       = "Object ID";
colVisitNew     = "검사 차수";
colDateNew      = "검사 날짜";
colBeamNew      = "nf(BEAM)";
colSensoryNew   = "np(Sensory)";
colAttentionNew = "na(Attention)";

% brainwave_dia_list
colSubject = "SUBJECT ID";
colDxDate  = "진단년월";
colDx      = "증세";

% snsb_total_fu
colSNSBObj   = "object_idx";
colSNSBVisit = "차수";
colZ         = "K_MMSE_total_score_z";

%% =========================================================
% 3. EEG 파일 유효성 필터링
%    - Object ID, 검사 차수, 검사 날짜, BEAM/Sensory/Attention이 모두 있어야 함
%    - 세 파일명은 ID로 시작하는 경우만 실제 파일로 간주
%% =========================================================
objText   = strtrim(string(EEG.(colObjNew)));
visitText = strtrim(string(EEG.(colVisitNew)));
dateText  = strtrim(string(EEG.(colDateNew)));

beamText = strtrim(string(EEG.(colBeamNew)));
sensText = strtrim(string(EEG.(colSensoryNew)));
attText  = strtrim(string(EEG.(colAttentionNew)));

hasBasicInfo = isValidText(objText) & ...
               isValidText(visitText) & ...
               isValidText(dateText) & ...
               isValidText(beamText) & ...
               isValidText(sensText) & ...
               isValidText(attText);

hasAllFiles = startsWith(beamText, "ID", "IgnoreCase", true) & ...
              startsWith(sensText, "ID", "IgnoreCase", true) & ...
              startsWith(attText,  "ID", "IgnoreCase", true);

EEG_valid = EEG(hasBasicInfo & hasAllFiles, :);

fprintf("전체 EEG row 수: %d\n", height(EEG));
fprintf("기본 정보 및 3개 EEG 파일 모두 유효한 row 수: %d\n", height(EEG_valid));

%% =========================================================
% 4. EEG Object ID / 검사 차수 / 검사 날짜 정리
%% =========================================================
EEG_valid.ObjectID_base = str2double(strtrim(string(EEG_valid.(colObjNew))));

visitText = strtrim(string(EEG_valid.(colVisitNew)));
EEG_valid.List_VisitNum = nan(height(EEG_valid),1);

for i = 1:height(EEG_valid)
    tok = regexp(visitText(i), "(\d+)-(\d+)", "tokens");

    if ~isempty(tok)
        EEG_valid.List_VisitNum(i) = str2double(tok{1}{2});
    end
end

EEG_valid.EEG_YYYYMM = nan(height(EEG_valid),1);

for i = 1:height(EEG_valid)
    EEG_valid.EEG_YYYYMM(i) = parseDateToYYYYMM(EEG_valid.(colDateNew)(i));
end

validEEGDate = ~isnan(EEG_valid.ObjectID_base) & ...
               ~isnan(EEG_valid.EEG_YYYYMM);

EEG_valid = EEG_valid(validEEGDate, :);

fprintf("Object ID 및 EEG 검사날짜까지 유효한 row 수: %d\n", height(EEG_valid));

%% =========================================================
% 5. DIA / SNSB ID 정리
%% =========================================================
DIA.Subject_ObjectID = extractNumericSubjectID(DIA.(colSubject));

if hasSNSB
    SNSB.ObjectID_base = extractBaseObjectID(SNSB.(colSNSBObj));
    SNSB.VisitNum = str2double(string(SNSB.(colSNSBVisit)));
end

%% =========================================================
% 6. EEG별 진단 차수 매칭 및 strict progression label 생성
%% =========================================================
allCandidate = table();

for i = 1:height(EEG_valid)

    thisObj = EEG_valid.ObjectID_base(i);
    eegYYYYMM = EEG_valid.EEG_YYYYMM(i);

    if isnan(thisObj) || isnan(eegYYYYMM)
        continue;
    end

    %% -----------------------------------------------------
    % 6-1. DIA에서 해당 피험자 찾기
    %% -----------------------------------------------------
    diaRow = DIA(DIA.Subject_ObjectID == thisObj, :);

    if isempty(diaRow)
        continue;
    end

    dxDateText = string(diaRow.(colDxDate)(1));
    dxText     = string(diaRow.(colDx)(1));

    visitDateTable = parseVisitDateSequence(dxDateText);
    dxTable        = parseDiagnosisSequence(dxText);

    if height(visitDateTable) < 2 || height(dxTable) < 2
        continue;
    end

    %% -----------------------------------------------------
    % 6-2. EEG 검사년월과 가장 가까운 진단년월 찾기
    %% -----------------------------------------------------
    diffs = abs(monthDistance(eegYYYYMM, visitDateTable.YYYYMM));
    [minDiff, matchedPos] = min(diffs);

    if minDiff > maxMonthDiff
        continue;
    end

    currOriginalVisit = visitDateTable.OriginalVisit(matchedPos);
    uniqueVisits = unique(visitDateTable.OriginalVisit, "stable");

    currVisitPos = find(uniqueVisits == currOriginalVisit, 1);

    if isempty(currVisitPos)
        continue;
    end

    %% -----------------------------------------------------
    % 6-3. 이전/현재/이후 위치 설정
    %      - 이전이 없거나 이후가 없는 경우도 허용
    %      - 단, 유지 클래스는 반드시 prev/current/next 모두 필요
    %      - 악화 클래스는 boundary도 일부 허용
    %% -----------------------------------------------------
    hasPrev = currVisitPos > 1;
    hasNext = currVisitPos < numel(uniqueVisits);

    prevOriginalVisit = nan;
    nextOriginalVisit = nan;
    prevPos = nan;
    nextPos = nan;

    currCandidates = find(visitDateTable.OriginalVisit == currOriginalVisit);
    currPos = matchedPos;

    if hasPrev
        prevOriginalVisit = uniqueVisits(currVisitPos - 1);
        prevCandidates = find(visitDateTable.OriginalVisit == prevOriginalVisit);
        prevPos = prevCandidates(end);     % 이전 차수는 현재와 가장 가까운 마지막 기록
    end

    if hasNext
        nextOriginalVisit = uniqueVisits(currVisitPos + 1);
        nextCandidates = find(visitDateTable.OriginalVisit == nextOriginalVisit);
        nextPos = nextCandidates(1);       % 이후 차수는 현재와 가장 가까운 첫 기록
    end

    %% -----------------------------------------------------
    % 6-4. 진단 추출
    %% -----------------------------------------------------
    prevDxRaw = "";
    nextDxRaw = "";

    if hasPrev
        prevDxRaw = getDxBySeqAll(dxTable, visitDateTable.SeqAll(prevPos));
    end

    currDxRaw = getDxBySeqAll(dxTable, visitDateTable.SeqAll(currPos));

    if hasNext
        nextDxRaw = getDxBySeqAll(dxTable, visitDateTable.SeqAll(nextPos));
    end

    prevDx = normalizeDxStrict(prevDxRaw);
    currDx = normalizeDxStrict(currDxRaw);
    nextDx = normalizeDxStrict(nextDxRaw);

    %% -----------------------------------------------------
    % 6-5. Strict progression label 생성
    %% -----------------------------------------------------
    [strictLabel, strictSubtype, reason] = classifyStrictProgression( ...
        prevDx, currDx, nextDx, hasPrev, hasNext);

    if strictLabel == "Exclude"
        includeFinal = false;
    else
        includeFinal = true;
    end

    %% -----------------------------------------------------
    % 6-6. SNSB/MMSE z-score는 참고용으로 저장
    %      - 이번 라벨링에서는 MMSE 방향성으로 필터링하지 않음
    %% -----------------------------------------------------
    prevZ = nan;
    currZ = nan;
    nextZ = nan;

    if hasSNSB
        if hasPrev
            prevZ = getSNSBZ(SNSB, thisObj, prevOriginalVisit, colZ);
        end

        currZ = getSNSBZ(SNSB, thisObj, currOriginalVisit, colZ);

        if hasNext
            nextZ = getSNSBZ(SNSB, thisObj, nextOriginalVisit, colZ);
        end
    end

    slopePrevCurr = nan;
    slopeCurrNext = nan;
    slopePrevNext = nan;

    if hasPrev && ~isnan(prevZ) && ~isnan(currZ)
        slopePrevCurr = currZ - prevZ;
    end

    if hasNext && ~isnan(currZ) && ~isnan(nextZ)
        slopeCurrNext = nextZ - currZ;
    end

    if hasPrev && hasNext && ~isnan(prevZ) && ~isnan(nextZ)
        slopePrevNext = nextZ - prevZ;
    end

    %% -----------------------------------------------------
    % 6-7. 결과 저장
    %% -----------------------------------------------------
    newRow = table( ...
        thisObj, ...
        string(EEG_valid.(colVisitNew)(i)), ...
        EEG_valid.List_VisitNum(i), ...
        string(EEG_valid.(colDateNew)(i)), ...
        eegYYYYMM, ...
        string(EEG_valid.(colBeamNew)(i)), ...
        string(EEG_valid.(colSensoryNew)(i)), ...
        string(EEG_valid.(colAttentionNew)(i)), ...
        currOriginalVisit, ...
        minDiff, ...
        hasPrev, hasNext, ...
        prevOriginalVisit, currOriginalVisit, nextOriginalVisit, ...
        prevDxRaw, currDxRaw, nextDxRaw, ...
        prevDx, currDx, nextDx, ...
        strictLabel, ...
        strictSubtype, ...
        reason, ...
        includeFinal, ...
        prevZ, currZ, nextZ, ...
        slopePrevCurr, slopeCurrNext, slopePrevNext, ...
        'VariableNames', ...
        ["ObjectID", ...
         "EEG_List_Visit_Text", ...
         "EEG_List_VisitNum", ...
         "EEG_Date", ...
         "EEG_YYYYMM", ...
         "BEAM_File", ...
         "Sensory_File", ...
         "Attention_File", ...
         "Matched_Original_Visit", ...
         "Month_Difference", ...
         "HasPrev", ...
         "HasNext", ...
         "Prev_Visit", ...
         "Current_Visit", ...
         "Next_Visit", ...
         "Prev_Diagnosis_Raw", ...
         "Current_Diagnosis_Raw", ...
         "Next_Diagnosis_Raw", ...
         "Prev_Diagnosis", ...
         "Current_Diagnosis", ...
         "Next_Diagnosis", ...
         "Strict_Label", ...
         "Strict_Subtype", ...
         "Exclude_Reason", ...
         "Include_Final", ...
         "Prev_K_MMSE_z", ...
         "Current_K_MMSE_z", ...
         "Next_K_MMSE_z", ...
         "Slope_Prev_to_Current", ...
         "Slope_Current_to_Next", ...
         "Slope_Prev_to_Next"]);

    allCandidate = [allCandidate; newRow];
end

%% =========================================================
% 7. 최종 리스트 생성
%% =========================================================
finalList = allCandidate(allCandidate.Include_Final == true, :);

%% 기존 파일 삭제 후 저장
if isfile(outAll)
    delete(outAll);
end

if isfile(outFinal)
    delete(outFinal);
end

writetable(allCandidate, outAll);
writetable(finalList, outFinal);

%% =========================================================
% 8. 요약 출력
%% =========================================================
fprintf("\n\n=== Strict progression 전체 후보 요약 ===\n");
fprintf("전체 후보 row 수: %d\n", height(allCandidate));
fprintf("전체 후보 피험자 수: %d명\n", numel(unique(allCandidate.ObjectID)));

fprintf("\nStrict_Label 분포 전체:\n");
disp(countByString(allCandidate.Strict_Label, "Strict_Label"));

fprintf("\nExclude reason 분포:\n");
disp(countByString(allCandidate.Exclude_Reason, "Exclude_Reason"));

fprintf("\n\n=== 최종 사용 리스트 요약 ===\n");
fprintf("최종 EEG 수: %d개\n", height(finalList));
fprintf("최종 피험자 수: %d명\n", numel(unique(finalList.ObjectID)));

fprintf("\n최종 Strict_Label 분포:\n");
disp(countByString(finalList.Strict_Label, "Strict_Label"));

fprintf("\n최종 Strict_Subtype 분포:\n");
disp(countByString(finalList.Strict_Subtype, "Strict_Subtype"));

fprintf("\n최종 현재 진단 분포:\n");
disp(countByString(finalList.Current_Diagnosis, "Current_Diagnosis"));

fprintf("\n저장 완료:\n");
fprintf("- %s\n", outAll);
fprintf("- %s\n", outFinal);

function T = readTablePreserve(fileName)

    opts = detectImportOptions(fileName);

    try
        opts.VariableNamingRule = 'preserve';
    catch
    end

    T = readtable(fileName, opts);
end


function mask = isValidText(x)

    x = strtrim(string(x));

    invalid = ["", "NaN", "nan", "<missing>", "missing", "없음", "진단중"];

    mask = ~ismissing(x) & ~ismember(x, invalid);
end


function id = extractNumericSubjectID(x)

    x = string(x);
    id = nan(size(x));

    for i = 1:numel(x)
        tok = regexp(x(i), "(\d+)", "tokens");

        if ~isempty(tok)
            id(i) = str2double(tok{end}{1});
        end
    end
end


function id = extractBaseObjectID(x)

    x = string(x);
    id = nan(size(x));

    for i = 1:numel(x)

        xi = strtrim(x(i));

        tok = regexp(xi, "^(\d+)", "tokens");

        if ~isempty(tok)
            id(i) = str2double(tok{1}{1});
        else
            val = str2double(xi);

            if ~isnan(val)
                id(i) = val;
            end
        end
    end
end


function yyyymm = parseDateToYYYYMM(x)

    yyyymm = nan;
    s = strtrim(string(x));

    if s == "" || ismissing(s)
        return;
    end

    % datetime 형태 또는 "2018-07-17"
    try
        dt = datetime(s);
        yyyymm = year(dt) * 100 + month(dt);
        return;
    catch
    end

    % "2018-07-17" 직접 처리
    tok = regexp(s, "(\d{4})[-/.](\d{1,2})", "tokens");

    if ~isempty(tok)
        y = str2double(tok{1}{1});
        m = str2double(tok{1}{2});
        yyyymm = y*100 + m;
        return;
    end

    % "201807"
    tok = regexp(s, "(\d{4})(\d{2})", "tokens");

    if ~isempty(tok)
        y = str2double(tok{1}{1});
        m = str2double(tok{1}{2});
        yyyymm = y*100 + m;
        return;
    end
end


function d = monthDistance(a, b)

    ay = floor(a / 100);
    am = mod(a, 100);

    by = floor(b / 100);
    bm = mod(b, 100);

    d = (ay - by) * 12 + (am - bm);
end


function visitTable = parseVisitDateSequence(s)

    s = string(s);

    tok = regexp(s, "(\d+)\s*차\s*:\s*(\d{6})", "tokens");

    n = numel(tok);

    SeqAll = (1:n)';
    OriginalVisit = nan(n,1);
    YYYYMM = nan(n,1);

    for i = 1:n
        OriginalVisit(i) = str2double(tok{i}{1});
        YYYYMM(i) = str2double(tok{i}{2});
    end

    visitTable = table(SeqAll, OriginalVisit, YYYYMM);
end


function dxTable = parseDiagnosisSequence(s)

    s = string(s);

    % 예: "1차 : CN, 2차 : MCI, 3차 : DEM"
    tok = regexp(s, "(\d+)\s*차\s*:\s*([^,;/]+)", "tokens");

    n = numel(tok);

    SeqAll = (1:n)';
    OriginalVisit = nan(n,1);
    Diagnosis = strings(n,1);

    for i = 1:n
        OriginalVisit(i) = str2double(tok{i}{1});
        Diagnosis(i) = strtrim(string(tok{i}{2}));
    end

    dxTable = table(SeqAll, OriginalVisit, Diagnosis);
end


function dx = getDxBySeqAll(dxTable, seqAll)

    dx = "";

    if isempty(dxTable) || isnan(seqAll)
        return;
    end

    idx = find(dxTable.SeqAll == seqAll, 1);

    if isempty(idx)
        return;
    end

    dx = string(dxTable.Diagnosis(idx));
end


function dx = normalizeDxStrict(x)

    x = upper(strtrim(string(x)));

    if x == "" || ismissing(x)
        dx = "Unknown";
        return;
    end

    if contains(x, "진단중") || contains(x, "없음") || ...
       contains(x, "UNKNOWN") || contains(x, "NAN") || ...
       contains(x, "MISSING")
        dx = "Unknown";
        return;
    end

    if x == "CN" || contains(x, "NORMAL")
        dx = "CN";

    elseif x == "SCD"
        dx = "SCD";

    elseif contains(x, "MCI")
        dx = "MCI";

    elseif x == "AD" || x == "D" || x == "DEM" || ...
           contains(x, "DEMENTIA") || contains(x, "DEM/D") || ...
           contains(x, "AD/D")
        dx = "DEM/D";

    elseif x == "NOS"
        dx = "NOS";

    else
        dx = "Unknown";
    end
end


function [label, subtype, reason] = classifyStrictProgression(prevDx, currDx, nextDx, hasPrev, hasNext)

    label = "Exclude";
    subtype = "Exclude";
    reason = "";

    invalid = ["", "Unknown", "NOS", "진단중", "없음", "<missing>", "NaN"];

    %% 현재 진단은 반드시 유효해야 함
    if ismember(currDx, invalid)
        reason = "Invalid current diagnosis";
        return;
    end

    %% -----------------------------------------------------
    % 1. 유지 클래스
    %    - 반드시 prev/current/next 모두 있어야 함
    %    - CN-CN-CN 또는 SCD-SCD-SCD만 허용
    %% -----------------------------------------------------
    if hasPrev && hasNext

        if ismember(prevDx, invalid) || ismember(nextDx, invalid)
            reason = "Invalid prev or next diagnosis";
            return;
        end

        if prevDx == currDx && currDx == nextDx && ...
           (currDx == "CN" || currDx == "SCD")

            label = "Maintain";
            subtype = currDx + "_Stable";
            reason = "Accepted: stable CN/SCD across three visits";
            return;
        end
    end

    %% -----------------------------------------------------
    % 2. 악화 클래스
    %    - 이용 가능한 adjacent pair가 모두 유지 또는 악화여야 함
    %    - 하나 이상의 악화 transition이 있어야 함
    %    - CN/SCD -> MCI/DEM 허용
    %    - MCI -> DEM 허용
    %    - DEM 이후 유지 DEM은 허용
    %    - 개선/회복/오락가락은 제외
    %% -----------------------------------------------------
    seq = strings(0,1);

    if hasPrev
        if ismember(prevDx, invalid)
            reason = "Invalid previous diagnosis";
            return;
        end
        seq(end+1,1) = prevDx;
    end

    seq(end+1,1) = currDx;

    if hasNext
        if ismember(nextDx, invalid)
            reason = "Invalid next diagnosis";
            return;
        end
        seq(end+1,1) = nextDx;
    end

    if numel(seq) < 2
        reason = "Not enough visits";
        return;
    end

    [isClearDecline, declineSubtype, declineReason] = isStrictDeclineSequence(seq);

    if isClearDecline
        label = "Declining";
        subtype = declineSubtype;
        reason = declineReason;
        return;
    end

    %% -----------------------------------------------------
    % 3. 나머지는 전부 제외
    %% -----------------------------------------------------
    reason = declineReason;
end


function [tf, subtype, reason] = isStrictDeclineSequence(seq)

    tf = false;
    subtype = "Exclude";
    reason = "Not strict decline";

    n = numel(seq);

    hasWorsening = false;
    transitions = strings(0,1);

    for i = 1:n-1

        a = seq(i);
        b = seq(i+1);

        if a == b
            transitions(end+1,1) = a + "_to_" + b;
            continue;
        end

        if isBaseNormal(a) && (b == "MCI" || b == "DEM/D")
            hasWorsening = true;
            transitions(end+1,1) = a + "_to_" + b;
            continue;
        end

        if a == "MCI" && b == "DEM/D"
            hasWorsening = true;
            transitions(end+1,1) = a + "_to_" + b;
            continue;
        end

        % 여기로 오면 개선 또는 애매한 변화
        % 예: MCI->CN, DEM->MCI, CN->SCD, SCD->CN 등
        reason = "Excluded: ambiguous, oscillating, or improving transition (" + ...
                 strjoin(seq, "_to_") + ")";
        return;
    end

    if ~hasWorsening
        reason = "Excluded: no worsening transition (" + strjoin(seq, "_to_") + ")";
        return;
    end

    tf = true;
    subtype = strjoin(transitions, "__");
    reason = "Accepted: strict worsening sequence (" + strjoin(seq, "_to_") + ")";
end


function tf = isBaseNormal(dx)

    tf = dx == "CN" || dx == "SCD";
end


function z = getSNSBZ(SNSB, objID, visitNum, colZ)

    z = nan;

    if isnan(objID) || isnan(visitNum)
        return;
    end

    rows = SNSB(SNSB.ObjectID_base == objID & SNSB.VisitNum == visitNum, :);

    if isempty(rows)
        return;
    end

    vals = str2double(string(rows.(colZ)));
    vals = vals(isfinite(vals));

    if isempty(vals)
        return;
    end

    z = mean(vals, "omitnan");
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