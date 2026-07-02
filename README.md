# Project 2 ASR: AISHELL-1 LVCSR and Whisper Fine-tuning

This repository contains the reproducible code, configuration, report source, and figures for Project 2.

Large generated artifacts are intentionally not committed:

- AISHELL-1 wav files
- Kaldi feature archives
- Kaldi experiment directories
- Whisper checkpoints and decoding outputs
- local Kaldi source/build directory

Main files:

- `Project2_ASR_Report.tex`: Chinese LaTeX report.
- `report_figures/`: figures used by the report.
- `Requirement.md`: assignment requirement.
- `project2_asr/`: Kaldi recipe and Whisper experiment code.
- `project2_asr/README.txt`: detailed reproduction notes and final results.

Final reported results:

| System | Test CER |
| --- | ---: |
| GMM-HMM best tri4a | 20.89% |
| Kaldi nnet3 TDNN | 17.39% |
| Whisper-small pretrained baseline | 26.77% |
| Whisper-small fine-tuned final + normalized scoring | 8.45% |

To reproduce, prepare AISHELL-1 data as described in `project2_asr/README.txt`, then run the Kaldi or Whisper scripts from `project2_asr/`.
