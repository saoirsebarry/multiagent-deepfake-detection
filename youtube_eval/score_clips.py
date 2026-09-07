"""Score extracted YouTube clips with the released system, agent by agent.

    python youtube_eval/score_clips.py clip_list.txt out.csv

clip_list.txt holds one npz path per line (the files extract_clips.py writes).
Runs src/detect_youtube.py's own loaders and per-agent analysis functions
under its default CONFIG (checkpoints/), on CPU, so a shared clip reproduces
its column in paper_artifacts/source_csvs/analysis_results_youtube.csv.
Report-stage libraries (dlib, LLM clients) are stubbed: only scores are needed.
"""
import csv
import logging
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name, attrs in {
    "dlib": [], "groq": ["Groq"], "transformers": ["AutoProcessor", "AutoModelForSpeechSeq2Seq"],
    "langchain_core": [], "langchain_core.runnables": ["RunnableLambda", "RunnableParallel"],
    "google": [], "google.generativeai": ["configure", "GenerativeModel"],
}.items():
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, object)
    sys.modules.setdefault(name, mod)
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import detect_youtube as dy  # noqa: E402

dy.CONFIG["device"] = "cpu"
dy.CONFIG["gradcam_output_dir"] = os.path.join(REPO, "youtube_eval", "xai_out")
os.makedirs(dy.CONFIG["gradcam_output_dir"], exist_ok=True)
logging.getLogger().setLevel(logging.WARNING)

AGENTS = [
    ("score_Visual (Spatial)", dy.run_visual_analysis),
    ("score_Audio (Mel+CNN)", dy.run_audio_analysis),
    ("score_Audio Forensics (ECAPA)", dy.run_audio_forensics_analysis),
    ("score_Cross-Modal (Lip-Sync)", dy.run_cross_modal_analysis),
    ("score_Facial Biometric (Quality)", dy.run_face_quality_analysis),
]


def main(list_path, out_csv):
    models = dy.load_all_models()
    missing = [k for k in ("spatial", "audio", "audio_forensics", "cross_modal", "face_quality",
                           "speaker_encoder", "audio_forensics_stats", "shap_explainer") if k not in models]
    if missing:
        raise SystemExit(f"models failed to load: {missing}")
    files = [line.strip() for line in open(list_path) if line.strip()]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filepath"] + [name for name, _ in AGENTS])
        for i, path in enumerate(files, 1):
            media = dy.debug_file_loading(path)
            row = [os.path.basename(path)]
            for _, fn in AGENTS:
                result = fn(media, models)
                score = result.get("score")
                row.append(f"{score:.6f}" if score is not None else "")
            writer.writerow(row)
            fh.flush()
            print(f"[{i}/{len(files)}] {os.path.basename(path)[:60]}", flush=True)
    print("SCORED ->", out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
