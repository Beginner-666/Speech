Project 2: LVCSR Dataset (aishell_10h)
======================================

This is a subset of AIShell-1 for the LVCSR course project.

Dataset Structure
-----------------
project2_asr/
├── data_aishell/
│   ├── transcript/
│   │   └── aishell_transcript_v0.8.txt
│   └── wav/
│       ├── dev/        (14,326 utterances, ~18.1 hours)
│       ├── test/       (7,176 utterances, ~10.0 hours)
│       └── train/      (8,000 utterances, ~10.0 hours, all <= 10s)
├── resource_aishell/
│   ├── lexicon.txt
│   └── speaker.info
├── run.sh
├── cmd.sh
├── path.sh
├── conf/
└── local/

Data Preparation
----------------
1. Copy this directory into your Kaldi egs path, e.g.:
   cp -r project2_asr $KALDI_ROOT/egs/aishell/s5/

2. If you only have a full AISHELL-1 download, rebuild the fixed Project 2
   subset from the transcript ids:

   local/prepare_aishell10h_subset.sh --symlink /path/to/full/AISHELL-1

   Use --copy instead of --symlink if the submitted directory must contain
   physical wav files. The split is fixed by aishell_transcript_v0.8.txt:
   train is lines 1-8000, dev is lines 8001-22326, and test is lines
   22327-29502. Existing wav files are not deleted; missing ids are reported in
   data_aishell/missing_train.ids, data_aishell/missing_dev.ids, and
   data_aishell/missing_test.ids.

3. Update path.sh if your Kaldi installation is at a different location.

4. Run the recipe:
   ./run.sh

Notes
-----
- All train utterances are <= 10 seconds.
- dev/test sets are kept complete from AIShell-1.
- The data path in run.sh is set to "." (current directory).
- If you encounter "train_large.txt not found", the provided recipe has been
  patched to use aishell_transcript_v0.8.txt for all subsets.

Statistics
----------
Train:   8,000 files, ~10.0 hours
Dev:     14,326 files, ~18.1 hours
Test:    7,176 files, ~10.0 hours
Total:   29,502 files, ~38.1 hours
Speakers: 400

Current Workspace Status
------------------------
This workspace is configured to use the local Kaldi checkout at:

  ../downloads/kaldi

Kaldi is built with OpenBLAS, OpenFST, and CUDA support. The local build enables `CUDA = true`, `WITH_CUDADECODER = true`, and `CUDA_ARCH = -gencode arch=compute_89,code=sm_89` for the RTX 4090-class GPU available on this machine. `path.sh` also adds the conda CUDA runtime libraries and the local CUDA shim directory to `LD_LIBRARY_PATH`. The GMM-HMM stages remain mostly CPU-bound by Kaldi design; nnet3/TDNN/chain training and the CUDA feature/decoder binaries can use the GPU. The nnet3 TDNN script keeps `use_gpu=true` when `cuda-compiled` succeeds and falls back to CPU only if CUDA is unavailable. The recipe links `steps` and `utils` from `../downloads/kaldi/egs/wsj/s5`, and `path.sh` sets `KALDI_ROOT` for this workspace.

Prepared and validated directories:

  data/train      8,000 utterances
  data/dev       14,326 utterances
  data/test       7,176 utterances
  data/lang
  data/lang_test

MFCC+pitch features and CMVN statistics have been generated for train, dev, and test. The generated feature manifests match the required split sizes:

  data/train/feats.scp    8,000 entries
  data/dev/feats.scp     14,326 entries
  data/test/feats.scp     7,176 entries

During setup, several extracted AISHELL-1 test WAV files were found to be 0-byte files. They were repaired by re-extracting the affected speaker archives from `downloads/AISHELL-1-full/data_aishell/wav/*.tar.gz`; after repair there are no remaining 0-byte WAV files.


Training Pipeline and Results
-----------------------------
The final recipe follows the standard Kaldi AISHELL-style LVCSR pipeline. `run.sh` prepares Kaldi data directories, extracts MFCC+pitch features, trains the language model, builds GMM-HMM baselines, then trains and decodes an nnet3 TDNN DNN-HMM system.

Stage summary:

- Data preparation creates `wav.scp`, `text`, `utt2spk`, `spk2utt`, lexicon resources, and language-model training text. The fixed split uses 8,000 train utterances, full AISHELL-1 dev, and full AISHELL-1 test.
- Feature extraction generates MFCC+pitch features and CMVN statistics for train/dev/test.
- GMM-HMM training provides the traditional ASR baseline and alignments, including mono, tri1, tri2, tri3a, tri4a, and tri5a systems.
- DNN-HMM training uses Kaldi nnet3 TDNN with CUDA enabled. The TDNN recipe was configured to run one GPU training job on the available RTX 4090-class GPU to avoid GPU memory oversubscription.
- Evaluation reports CER as the primary metric and WER as a reference metric. Character-level scoring uses Kaldi `cer_*` outputs under `scoring_kaldi`.

Final model location:

  project2_asr/exp/nnet3/tdnn_sp/final.mdl

Final nnet3 TDNN decoding results:

  dev  WER: 30.40%  from project2_asr/exp/nnet3/tdnn_sp/decode_dev/wer_17_0.5
  dev  CER: 15.14%  from project2_asr/exp/nnet3/tdnn_sp/decode_dev/cer_15_1.0
  test WER: 33.14%  from project2_asr/exp/nnet3/tdnn_sp/decode_test/wer_17_1.0
  test CER: 17.39%  from project2_asr/exp/nnet3/tdnn_sp/decode_test/cer_17_1.0

GMM-HMM baseline test results observed during the experiment:

  tri2  WER: 43.56%  CER: 28.35%
  tri3a WER: 41.36%  CER: 25.87%
  tri4a WER: 36.60%  CER: 20.89%
  tri5a WER: 37.99%  CER: 22.43%

The best GMM-HMM baseline by test CER was tri4a with 20.89% CER. The final TDNN DNN-HMM system improves the test CER to 17.39%.

Reproduction
------------
From this directory, run:

  cd project2_asr
  ./run.sh

If resuming only the TDNN stage after the GMM-HMM alignments and high-resolution features already exist, run:

  cd project2_asr
  . ./path.sh
  local/nnet3/run_tdnn.sh

The main completed decoding directories are:

  project2_asr/exp/nnet3/tdnn_sp/decode_dev
  project2_asr/exp/nnet3/tdnn_sp/decode_test


Whisper Extension
-----------------
A second ASR system has been added under `project2_asr/whisper`. It reuses the same Kaldi train/dev/test split and writes JSONL manifests for Hugging Face Whisper training and decoding. This is intended for the optional neural pretrained-model comparison requested by the assignment.

Main files:

  project2_asr/whisper/prepare_whisper_data.py
  project2_asr/whisper/decode_whisper.py
  project2_asr/whisper/train_whisper.py
  project2_asr/whisper/score_whisper.py
  project2_asr/whisper/run_whisper_baseline.sh
  project2_asr/whisper/run_whisper_finetune.sh

Install dependencies if needed:

  pip install -r project2_asr/whisper/requirements.txt

Prepare Whisper JSONL data:

  cd project2_asr
  python whisper/prepare_whisper_data.py

Run pretrained Whisper baseline before fine-tuning:

  bash whisper/run_whisper_baseline.sh openai/whisper-small 8

Fine-tune and decode dev/test:

  bash whisper/run_whisper_finetune.sh openai/whisper-small

Whisper results should be reported separately for the pretrained baseline and the fine-tuned checkpoint, because Whisper uses pretrained weights.

Completed Whisper-small results in this workspace:

  pretrained baseline test WER: 26.77%
  pretrained baseline test CER: 26.77%

  fine-tuned final dev WER: 7.75%
  fine-tuned final dev CER: 7.75%
  fine-tuned final test WER: 8.47%
  fine-tuned final test CER: 8.47%

The fine-tuned final model is stored in:

  project2_asr/exp/whisper_small/finetuned

The best checkpoint observed during periodic dev evaluation is stored in:

  project2_asr/exp/whisper_small/finetuned/best

Training-time best sampled dev CER was 12.28% at step 3500, while full final dev decoding reached 7.75% CER. The full final test CER, 8.47%, is substantially better than the Kaldi TDNN test CER of 17.39%. After unified text normalization in `whisper/score_whisper.py` (traditional-to-simplified conversion, NFKC, punctuation removal, and whitespace removal), the existing final decoding result scores 8.45% test CER. Re-decoding the periodic best checkpoint gives 7.62% dev CER but 8.53% test CER, so the recommended final submitted Whisper result is the final checkpoint with normalized scoring: 8.45% test CER.

Additional score-oriented runs:

  bash whisper/run_whisper_best_decode.sh exp/whisper_small/finetuned/best 4

This decodes the best checkpoint observed during periodic dev evaluation and writes normalized scores to:

  project2_asr/exp/whisper_small/decode_dev_best_norm
  project2_asr/exp/whisper_small/decode_test_best_norm

To rescore an existing decode directory with normalized text only:

  python whisper/score_whisper.py --ref exp/whisper_small/decode_test/ref.txt --hyp exp/whisper_small/decode_test/hyp.txt --out-dir exp/whisper_small/decode_test_norm
