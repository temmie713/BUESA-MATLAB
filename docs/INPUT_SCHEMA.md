# Input schema

This repository does not distribute participant-level GARD data. The scripts expect locally authorized source tables and EEG files.

## EEG visit/file list

`01_construct_longitudinal_labels.m` expects the source fields:

- `Object ID`
- `검사 차수`
- `검사 날짜`
- `nf(BEAM)`
- `np(Sensory)`
- `na(Attention)`

The three acquisition-file fields are used to identify protocol-complete EEG visits. Only the BEAM recording is processed by the feature-extraction script.

## Longitudinal diagnosis table

Expected fields:

- `SUBJECT ID`
- `진단년월`
- `증세`

The code parses the longitudinal visit dates and diagnosis sequence and matches each EEG visit to the nearest eligible diagnostic assessment within the configured time window.

## Optional cognitive-score table

Optional fields:

- `object_idx`
- `차수`
- `K_MMSE_total_score_z`

These values are not mandatory criteria for the primary class assignment in the public workflow.

## BEAM EEG text files

`02_extract_two_channel_eeg_features.m` recursively searches the configured `beamRoot` for `.txt` files and matches them to the BEAM file names stored in the strict progression list.

The input signal must contain at least two numeric EEG channels. The public analysis uses channels 1 and 2 at 250 Hz.

## Generated feature table

The main feature-extraction output used by scripts 03-06 is:

`AllData_FileLevelFeatures.xlsx`

It contains metadata/label fields plus the 64 numeric EEG features.

## Privacy

Do not add real participant-level tables, raw EEG, ObjectIDs, diagnosis histories, or identifiable local file paths to the public repository. Use synthetic examples if an executable demonstration dataset is added later.
