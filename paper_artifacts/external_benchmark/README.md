# External benchmark artefacts (MAVOS-DD)

`mavos_dd/manifest.csv` lists every clip of the stratified test sample (paths relative to the MAVOS-DD release);
`metadata.csv` and `modality.csv` carry the release's generator, language, source-video and forged-modality labels.
`original/`, `crf23/`, `crf40/` hold the per-clip agent scores of the released five-agent system for each compression tier
(`tools/external_benchmark/run_chunked.py`); `report_original.json` is `report_by_group.py` on the original tier;
`fusion_recalibration.json` is `fusion_recalibration.py`; `added_agent/` holds the added EfficientNet-B0 agent's training and
validation manifests, training log and test scores (`train_added_agent.py`, `score_added_agent.py`), and `added_agent.json`
is `evaluate_added_agent.py`. The added agent's checkpoint (16 MB) is not committed; it retrains in about 30 minutes.
