# ===========================================================================
# BLOCK A - MUST be the FIRST executable lines of
#           src/data_preprocessing/preprocessed_all_unbalanced.py,
#           ABOVE every other import (TensorFlow reads these at import time).
# ===========================================================================
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import os
import shutil
import json
import cv2
import numpy as np
import librosa
# MTCNN is imported inside each worker (_build_detector); a module-scope
# import would load TensorFlow before a spawned worker can pin its threads.
import argparse
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = 'polyglot_lang'


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ===========================================================================
# BLOCK B - replaces process_and_save_files (lines 19-115 of the original).
#
# Three other one-line edits are required in the same file:
#   1. delete the module-scope `from mtcnn.mtcnn import MTCNN` (line 7).
#      Under spawn every worker re-imports this module, and a module-scope
#      TensorFlow import loads TF before the worker can pin its thread counts.
#      _build_detector() imports it inside the worker instead.
#   2. in main(), `detector = MTCNN()`  ->  `detector = None`.
#      The parent must not hold a CUDA context the workers want.
#   3. in the __main__ block, call `add_parallel_args(parser)` next to the
#      other add_argument calls.
# The existing `if __name__ == '__main__':` guard is load-bearing under spawn -
# do not move main() or parse_args() above it.
# ===========================================================================
import contextlib
import errno
import io
import multiprocessing as mp
import shutil
import signal
import sys
import time
import uuid
import zipfile

import cv2
import librosa
import numpy as np
from tqdm import tqdm

_PARTIAL_SUFFIX = ".partial"
_STALE_PARTIAL_SECONDS = 3600.0
_DISK_CHECK_SECONDS = 30.0
_MAX_CONSECUTIVE_FAILURES = 5
_MAX_FAILURE_FRACTION = 0.02

_HARD_FAILURES = ("error_save", "error_worker", "fatal_init")
_FSYNC_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP,
                      errno.EBADF, errno.EPERM}

_WORKER = {"detector": None, "cfg": None, "fatal": None}


def add_parallel_args(parser):
    """Register the parallel flags. Every flag has a safe default and the code
    below falls back to the serial reference path when they are absent."""
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker processes for clip extraction. 4 is the suggested "
                             "start on an 8 vCPU Colab box; raise to 6 only if nvidia-smi "
                             "shows GPU memory headroom. 1 or less runs the original "
                             "in-process serial loop, which is the reference path.")
    parser.add_argument("--detector_device", choices=("auto", "cpu"), default="auto",
                        help="'auto' leaves MTCNN on whatever device TensorFlow picks, "
                             "which is what the serial run used. 'cpu' hides the GPU from "
                             "the workers; it can change crops by a pixel, so only use it "
                             "if you are prepared to regenerate the whole dataset.")
    parser.add_argument("--stall_timeout", type=float, default=900.0,
                        help="Abort if no clip finishes for this many seconds. Turns a "
                             "pool hung by an OOM-killed worker into a clear, resumable "
                             "error instead of dead GPU time.")
    parser.add_argument("--verify_writes", action="store_true",
                        help="Additionally np.load() each staged .npz before renaming it "
                             "into place. The cheap size + zip-directory checks always run.")
    parser.add_argument("--verify_existing", action="store_true",
                        help="Validate the .npz already in the output directory and delete "
                             "any that are truncated, so they are rebuilt. Run this once to "
                             "clean up after interrupted runs of the old serial code.")
    parser.add_argument("--verbose_saves", action="store_true",
                        help="Print the original per-clip '-> SAVING' line. Off by default; "
                             "the progress bar and the end-of-split summary replace it.")
    return parser


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def _resolve_workers(args):
    """Absent flag -> 1 -> the serial reference path. Parallelism is opt-in."""
    try:
        return int(getattr(args, "workers", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _resolve_min_free_gib(args):
    value = getattr(args, "min_free_gib", 1.0)
    return 1.0 if value is None else float(value)


def _worker_cfg(args, workers):
    """Plain picklable dict. Attribute errors surface here, in the parent, loudly."""
    cpu = os.cpu_count() or 2
    return {
        "image_size": args.image_size,
        "max_faces": args.max_faces,
        "frame_stride": args.frame_stride,
        "sample_rate": args.sample_rate,
        "detector_device": getattr(args, "detector_device", "auto"),
        "verify_writes": bool(getattr(args, "verify_writes", False)),
        "verbose_saves": bool(getattr(args, "verbose_saves", False)),
        "tf_threads": max(1, cpu // max(1, workers)),
    }


def _output_path(output_dir, video_path, label):
    """The serial naming exactly, including .replace() rather than a suffix strip."""
    video_basename = os.path.basename(video_path).replace(".mp4", "")
    return os.path.join(
        output_dir, f"{video_basename}_label_{'fake' if int(label) == 1 else 'real'}.npz")


# ---------------------------------------------------------------------------
# atomic write
# ---------------------------------------------------------------------------
def _fsync_fd(fd):
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _FSYNC_UNSUPPORTED:
            raise


def _fsync_dir(directory):
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        _fsync_fd(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_savez(output_filename, faces, waveform, string_label, verify_load):
    """Compress in memory, write once, verify, then rename into place.

    The final name only ever appears complete, so an interrupted run leaves a
    dot-prefixed *.partial file that the resume scan ignores rather than a
    truncated .npz a later run would treat as done. Compressing in memory also
    collapses the dozens of small FUSE writes np.savez_compressed(path, ...)
    would otherwise dribble out into one write of one ~4 MiB file.
    """
    buffer = io.BytesIO()
    np.savez_compressed(buffer, faces=faces, waveform=waveform,
                        label=np.array([string_label]))
    payload = buffer.getbuffer()
    expected = payload.nbytes

    directory = os.path.dirname(output_filename) or "."
    base = os.path.basename(output_filename)
    tmp_path = os.path.join(directory, f".{base}.{uuid.uuid4().hex[:12]}{_PARTIAL_SUFFIX}")
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            _fsync_fd(handle.fileno())

        landed = os.path.getsize(tmp_path)
        if landed != expected:
            raise OSError(f"short write: {landed} of {expected} bytes reached {tmp_path}")
        if not zipfile.is_zipfile(tmp_path):
            raise OSError(f"staged npz failed its container check: {tmp_path}")
        if verify_load:
            with np.load(tmp_path, allow_pickle=False) as archive:
                for key in ("faces", "waveform", "label"):
                    archive[key]

        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, output_filename)
        tmp_path = None
        _fsync_dir(directory)
    finally:
        del payload
        buffer.close()
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# per-clip work - shared by the serial and the parallel path so they cannot drift
# ---------------------------------------------------------------------------
def _process_clip(video_path, label, output_filename, detector, cfg):
    """One clip, transcribed from the serial loop. Per-clip failures are recorded,
    never raised, so one bad clip costs one clip exactly as it did serially."""
    result = {"status": "written", "no_audio": False, "messages": []}
    video_basename = os.path.basename(video_path).replace(".mp4", "")

    faces = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened() and len(faces) < cfg["max_faces"]:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % cfg["frame_stride"] == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(frame_rgb)
                for det in detections:
                    if det["confidence"] > 0.95:
                        x1, y1, width, height = det["box"]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = x1 + width, y1 + height
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            resized_face = cv2.resize(
                                face_crop, (cfg["image_size"], cfg["image_size"]))
                            faces.append(resized_face)
                            if len(faces) >= cfg["max_faces"]:
                                break

            frame_count += 1
    except Exception as e:
        result["status"] = "error_frames"
        result["messages"].append(
            f"Warning: Could not process video frames from {video_basename}. Error: {e}")
        return result
    finally:
        # The serial loop leaked the handle when the body raised; a long-lived
        # worker cannot afford that.
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()

    if not faces:
        result["status"] = "skipped_no_faces"
        result["messages"].append(
            f"Warning: No faces found in {video_basename}. Skipping file save.")
        return result

    waveform = np.array([])
    audio_found = False
    try:
        loaded_waveform, _ = librosa.load(video_path, sr=cfg["sample_rate"], mono=True)
        if loaded_waveform is not None and loaded_waveform.size > 0:
            waveform = loaded_waveform
            audio_found = True
    except (AttributeError, NameError, TypeError, ImportError):
        # A bug here would otherwise be laundered into a silently empty waveform
        # on every clip, which is unrecoverable once the .npz exist.
        raise
    except Exception as e:
        result["messages"].append(
            f"Warning: Could not load audio for {video_basename}. Error: {e}. "
            f"Saving with empty audio.")

    if not audio_found:
        result["messages"].append(f"-> Logging {video_basename} for audio issue report.")
        result["no_audio"] = True

    string_label = "fake" if int(label) == 1 else "real"
    if cfg["verbose_saves"]:
        result["messages"].append(
            f"-> SAVING: '{output_filename}' with internal label: '{string_label}'")
    try:
        _atomic_savez(output_filename, np.array(faces), waveform, string_label,
                      cfg["verify_writes"])
    except Exception as e:
        result["status"] = "error_save"
        result["messages"].append(f"Error saving file {output_filename}. Error: {e}")
    return result


# ---------------------------------------------------------------------------
# worker process
# ---------------------------------------------------------------------------
def _build_detector(cfg):
    from mtcnn.mtcnn import MTCNN  # imported here so TF loads only inside the worker

    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(cfg["tf_threads"])
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

    detector = None  # built lazily per worker; see _build_detector
    # Warm up while the lock is held: TF defers the GPU allocation to the first
    # detect_faces, and N simultaneous allocations are what exhausts the device.
    detector.detect_faces(np.zeros((256, 256, 3), dtype=np.uint8))
    return detector


def _worker_init(cfg, init_lock):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent owns Ctrl-C
    _WORKER["cfg"] = cfg
    with contextlib.suppress(Exception):
        cv2.setNumThreads(1)
    try:
        with init_lock:
            _WORKER["detector"] = _build_detector(cfg)
    except BaseException as exc:
        # Never raise here: Pool would respawn this worker forever. The first
        # task reports it and the parent aborts the run.
        _WORKER["fatal"] = f"{type(exc).__name__}: {exc}"


def _worker_task(task):
    index, video_path, label, output_filename = task
    if _WORKER["fatal"] is not None:
        result = {"status": "fatal_init", "no_audio": False,
                  "messages": [f"FATAL: worker could not initialise MTCNN: "
                               f"{_WORKER['fatal']}"]}
    else:
        try:
            result = _process_clip(video_path, label, output_filename,
                                   _WORKER["detector"], _WORKER["cfg"])
        except BaseException as exc:
            result = {"status": "error_worker", "no_audio": False,
                      "messages": [f"Error: worker failed on "
                                   f"{os.path.basename(video_path)}: "
                                   f"{type(exc).__name__}: {exc}"]}
    result["index"] = index
    result["video_path"] = video_path
    return result


@contextlib.contextmanager
def _child_env(cfg):
    """Child processes inherit the environment as it stands at spawn time, so the
    thread pinning and any device hiding must be applied in the parent here and
    removed again afterwards."""
    overrides = {
        "OMP_NUM_THREADS": str(cfg["tf_threads"]),
        "OPENBLAS_NUM_THREADS": str(cfg["tf_threads"]),
        "MKL_NUM_THREADS": str(cfg["tf_threads"]),
        "NUMEXPR_NUM_THREADS": str(cfg["tf_threads"]),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    }
    if cfg["detector_device"] == "cpu":
        overrides["CUDA_VISIBLE_DEVICES"] = ""
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
def _sweep_stale_partials(output_dir):
    """Remove staging files left by an earlier killed run. They are never mistaken
    for finished clips; they only occupy space. Anything recent is left alone so a
    second concurrent run is not sabotaged."""
    cutoff = time.time() - _STALE_PARTIAL_SECONDS
    removed = 0
    freed = 0
    try:
        names = os.listdir(output_dir)
    except OSError:
        return
    for name in names:
        if not name.endswith(_PARTIAL_SUFFIX):
            continue
        path = os.path.join(output_dir, name)
        try:
            stat = os.stat(path)
            if stat.st_mtime >= cutoff:
                continue
            os.unlink(path)
        except OSError:
            continue
        removed += 1
        freed += stat.st_size
    if removed:
        print(f"  - removed {removed} stale staging file(s) ({freed / 2 ** 20:.0f} MiB) "
              f"from an earlier interrupted run")


def _verify_existing_outputs(output_dir, names, verify_load):
    """Delete .npz that are truncated so they are rebuilt. The old serial code wrote
    straight to the final name, so an interrupted serial run can have left one."""
    bad = 0
    for name in sorted(names):
        path = os.path.join(output_dir, name)
        ok = True
        try:
            if not zipfile.is_zipfile(path):
                ok = False
            elif verify_load:
                with np.load(path, allow_pickle=False) as archive:
                    for key in ("faces", "waveform", "label"):
                        archive[key]
        except Exception:
            ok = False
        if not ok:
            bad += 1
            with contextlib.suppress(OSError):
                os.unlink(path)
            tqdm.write(f"  - corrupt output removed, will be rebuilt: {name}")
    print(f"  - verified {len(names)} existing .npz, removed {bad}")
    return {name for name in names if os.path.exists(os.path.join(output_dir, name))}


def _report_output_name_collisions(file_list, output_dir):
    seen = {}
    collisions = 0
    for video_path, label in file_list:
        name = os.path.basename(_output_path(output_dir, video_path, label))
        if name in seen:
            collisions += 1
            if collisions <= 3:
                tqdm.write(f"  - WARNING: '{name}' is produced by both "
                           f"{seen[name]} and {video_path}")
        else:
            seen[name] = video_path
    if collisions:
        print(f"  - WARNING: {collisions} clip(s) share an output filename with an "
              f"earlier clip. Serial behaviour (first one wins, the later one is only "
              f"processed if the first wrote nothing) is reproduced, but the on-disk "
              f"count will be lower than the split size.")
    return collisions


def _free_gib(path):
    return shutil.disk_usage(path).free / 2 ** 30


def _disk_abort(free_gib, min_free_gib, written):
    return SystemExit(
        f"ABORT: only {free_gib:.2f} GiB free on the output volume "
        f"(--min_free_gib {min_free_gib}). {written} clips written this run; "
        f"free space and re-run to resume.")


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------
def _select_wave(entries, existing_names, output_dir, attempts_left):
    """The clips to dispatch next, in file_list order.

    Reproduces the serial --limit semantics exactly: serial breaks out of the loop
    once it has attempted `limit` clips, and skips (without spending budget) any
    clip whose .npz already exists. Two clips that map to the same output filename
    are never dispatched together - the second is deferred to the next wave, which
    is where serial would have seen the first one's file.
    """
    wave = []
    deferred = []
    claimed = set()
    limit_reached = False
    for index, video_path, label in entries:
        output_filename = _output_path(output_dir, video_path, label)
        name = os.path.basename(output_filename)
        if name in existing_names:
            continue
        if name in claimed:
            deferred.append((index, video_path, label))
            continue
        if attempts_left is not None and attempts_left <= 0:
            limit_reached = True
            break
        claimed.add(name)
        wave.append((index, video_path, label, output_filename))
        if attempts_left is not None:
            attempts_left -= 1
    return wave, deferred, attempts_left, limit_reached


class _RunState:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.counts = {}
        self.written = 0
        self.attempted = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.no_audio_by_index = {}
        self.written_names = set()
        self.fatal_message = None
        self.disk_abort = None
        self.stalled = False
        self.last_disk_check = 0.0


def _consume(result, state, output_dir, min_free_gib, now):
    for message in result["messages"]:
        tqdm.write(message)

    status = result["status"]
    state.counts[status] = state.counts.get(status, 0) + 1
    state.attempted += 1
    if status == "written":
        state.written += 1
        state.written_names.add(os.path.basename(
            result.get("output_filename") or ""))
    if result["no_audio"]:
        state.no_audio_by_index[result["index"]] = result["video_path"]

    if status in _HARD_FAILURES:
        state.failures += 1
        state.consecutive_failures += 1
        if status == "fatal_init" and state.fatal_message is None:
            state.fatal_message = result["messages"][0] if result["messages"] else status
    else:
        state.consecutive_failures = 0

    if now - state.last_disk_check >= _DISK_CHECK_SECONDS:
        state.last_disk_check = now
        free = _free_gib(output_dir)
        if free < min_free_gib and state.disk_abort is None:
            state.disk_abort = _disk_abort(free, min_free_gib, state.written)


def _should_stop(state):
    if state.fatal_message is not None or state.disk_abort is not None:
        return True
    if state.consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
        return True
    budget = max(_MAX_CONSECUTIVE_FAILURES,
                 int(state.total_tasks * _MAX_FAILURE_FRACTION))
    return state.failures > budget


def _run_wave_serial(wave, detector, cfg, state, output_dir, min_free_gib, bar):
    for index, video_path, label, output_filename in wave:
        result = _process_clip(video_path, label, output_filename, detector, cfg)
        result["index"] = index
        result["video_path"] = video_path
        result["output_filename"] = output_filename
        _consume(result, state, output_dir, min_free_gib, time.monotonic())
        bar.update(1)
        if _should_stop(state):
            break


def _run_wave_parallel(wave, workers, cfg, state, output_dir, min_free_gib,
                       stall_timeout, bar):
    tasks = [(index, video_path, label, output_filename)
             for index, video_path, label, output_filename in wave]
    by_index = {task[0]: task[3] for task in tasks}

    ctx = mp.get_context("spawn")
    with _child_env(cfg):
        init_lock = ctx.Lock()
        pool = ctx.Pool(processes=workers, initializer=_worker_init,
                        initargs=(cfg, init_lock))
        drained = False
        try:
            # chunksize=1: clips run from seconds to minutes, so a static shard
            # would end with idle workers waiting on one straggler. Each task
            # still goes to exactly one worker.
            results = pool.imap_unordered(_worker_task, tasks, chunksize=1)
            while True:
                try:
                    result = results.next(timeout=stall_timeout)
                except StopIteration:
                    drained = True
                    break
                except mp.TimeoutError:
                    # A worker killed by the OOM reaper never returns a result and
                    # Pool would otherwise wait forever, burning the session.
                    state.stalled = True
                    break
                result["output_filename"] = by_index.get(result["index"], "")
                _consume(result, state, output_dir, min_free_gib, time.monotonic())
                bar.update(1)
                if _should_stop(state):
                    break
        finally:
            if drained:
                pool.close()
            else:
                pool.terminate()
            pool.join()


# ---------------------------------------------------------------------------
# the drop-in replacement
# ---------------------------------------------------------------------------
def process_and_save_files(file_list, output_dir, detector, args):
    """
    Processes a list of video files, saves the data, and returns a list of any files
    that were found to have no audio.

    The per-clip work is distributed over a spawn-based process pool. `detector` is
    kept for signature compatibility and is used only when --workers <= 1; each
    worker builds its own MTCNN after spawn, because an initialised CUDA context
    cannot be inherited safely.
    """
    if not file_list:
        return []

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing {len(file_list)} files for the '{os.path.basename(output_dir)}' set...")

    workers = _resolve_workers(args)
    cfg = _worker_cfg(args, workers)
    limit = int(getattr(args, "limit", 0) or 0)
    min_free_gib = _resolve_min_free_gib(args)
    stall_timeout = float(getattr(args, "stall_timeout", 900.0) or 900.0)

    if workers > 1 and getattr(sys.modules.get("__main__"), "__file__", None) is None:
        raise SystemExit(
            "ABORT: --workers > 1 needs this script to be run as a file "
            "(`python src/data_preprocessing/preprocessed_all_unbalanced.py ...` or "
            "`!python ...` from a notebook cell). The spawn start method re-imports "
            "__main__ in every worker and cannot do that from an interactive session. "
            "Run it as a script, or pass --workers 1.")

    _sweep_stale_partials(output_dir)

    try:
        existing_names = {name for name in os.listdir(output_dir) if name.endswith(".npz")}
    except OSError:
        existing_names = set()
    if getattr(args, "verify_existing", False):
        existing_names = _verify_existing_outputs(
            output_dir, existing_names, cfg["verify_writes"])

    _report_output_name_collisions(file_list, output_dir)

    free_gib = _free_gib(output_dir)
    if free_gib < min_free_gib:
        raise _disk_abort(free_gib, min_free_gib, 0)

    entries = [(index, video_path, label)
               for index, (video_path, label) in enumerate(file_list)]
    attempts_left = limit if limit else None

    total_pending = sum(
        1 for _, video_path, label in entries
        if os.path.basename(_output_path(output_dir, video_path, label)) not in existing_names)
    state = _RunState(min(total_pending, limit) if limit else total_pending)

    print(f"  - {len(file_list) - total_pending} already written, {total_pending} to process"
          f"{f' (--limit {limit})' if limit else ''} on {workers} worker(s)")

    bar = tqdm(total=state.total_tasks,
               desc=f"Processing {os.path.basename(output_dir)}")
    serial_detector = detector
    limit_reached = False
    try:
        while entries:
            wave, deferred, attempts_left, limit_reached = _select_wave(
                entries, existing_names, output_dir, attempts_left)
            if not wave:
                break
            if workers <= 1:
                if serial_detector is None:
                    serial_detector = _build_detector(cfg)
                _run_wave_serial(wave, serial_detector, cfg, state, output_dir,
                                 min_free_gib, bar)
            else:
                _run_wave_parallel(wave, workers, cfg, state, output_dir, min_free_gib,
                                   stall_timeout, bar)
            if _should_stop(state) or state.stalled or limit_reached:
                break
            existing_names |= {name for name in state.written_names if name}
            entries = deferred
    finally:
        bar.close()

    if limit_reached:
        tqdm.write(f"-> reached --limit {limit}; re-run to continue")

    summary = ", ".join(f"{key}={value}" for key, value in sorted(state.counts.items()))
    print(f"  - {os.path.basename(output_dir)}: {summary or 'nothing processed'}")

    if state.fatal_message is not None:
        raise SystemExit(
            f"ABORT: {state.fatal_message}\n"
            f"{state.written} clips written this run; every finished clip is on disk, "
            f"so re-run to resume. If this is GPU memory, lower --workers.")
    if state.disk_abort is not None:
        raise state.disk_abort
    if state.stalled:
        raise SystemExit(
            f"ABORT: no clip completed for {stall_timeout:.0f}s, so a worker was most "
            f"likely killed (OOM). {state.attempted} of {state.total_tasks} clips were "
            f"accounted for and {state.written} written this run; re-run to resume, "
            f"ideally with a lower --workers.")
    if state.failures:
        raise SystemExit(
            f"ABORT: {state.failures} clip(s) failed to be written "
            f"({summary}). {state.written} clips written this run; nothing partial was "
            f"left on disk, so fix the cause and re-run to resume.")

    return [state.no_audio_by_index[index] for index in sorted(state.no_audio_by_index)]


def main(args):
    """Main function to orchestrate the data processing and splitting pipeline."""
    global DATA_DIR
    DATA_DIR = args.data_dir
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"ABORT: --data_dir not found: {DATA_DIR}")
    for sub in ('json_file', 'real', 'fake'):
        if not os.path.isdir(os.path.join(DATA_DIR, sub)):
            raise SystemExit(
                f"ABORT: {DATA_DIR} does not look like the PolyGlotFake release "
                f"(missing {sub}/). Expected json_file/, real/ and fake/.")

    LANGUAGES = ['ar', 'en', 'es', 'fr', 'ja', 'ru', 'zh']
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)


    combined_train_files = []
    combined_val_files = []
    combined_test_files = []

    for lang in LANGUAGES:
        print(f"\n---> Processing and splitting language: {lang.upper()}")
        lang_real_files = []
        lang_fake_files = []

        # Load REAL videos
        real_json_path = os.path.join(DATA_DIR, 'json_file', 'real_json_file', f'{lang}.json')
        if os.path.exists(real_json_path):
            with open(real_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            video_entries = data.get('video') or data.get('videos', []) if isinstance(data, dict) else data
            for entry in video_entries:
                filename = entry.get('filename') or entry.get('name')
                if not filename: continue
                full_path = os.path.join(DATA_DIR, 'real', lang, filename + ('' if filename.endswith('.mp4') else '.mp4'))
                if os.path.exists(full_path):
                    lang_real_files.append((full_path, 0))

        # Load fake videos
        fake_json_path = os.path.join(DATA_DIR, 'json_file', 'fake_Json_file', f'to_{lang}.json')
        if os.path.exists(fake_json_path):
            with open(fake_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            video_entries = data.get('video') or data.get('videos', []) if isinstance(data, dict) else data
            for entry in video_entries:
                filename = entry.get('filename') or entry.get('name')
                if not filename: continue
                full_path = os.path.join(DATA_DIR, 'fake', f'to_{lang}', filename + ('' if filename.endswith('.mp4') else '.mp4'))
                if os.path.exists(full_path):
                    lang_fake_files.append((full_path, 1))
        
        lang_real_files.sort()
        lang_fake_files.sort()
        

        # Split data
        if lang_real_files:
            real_train_val, real_test = train_test_split(lang_real_files, test_size=0.15, random_state=RANDOM_SEED)
            real_train, real_val = train_test_split(real_train_val, test_size=0.1765, random_state=RANDOM_SEED)
            combined_train_files.extend(real_train)
            combined_val_files.extend(real_val)
            combined_test_files.extend(real_test)

        if lang_fake_files:
            fake_train_val, fake_test = train_test_split(lang_fake_files, test_size=0.15, random_state=RANDOM_SEED)
            fake_train, fake_val = train_test_split(fake_train_val, test_size=0.1765, random_state=RANDOM_SEED)
            combined_train_files.extend(fake_train)
            combined_val_files.extend(fake_val)
            combined_test_files.extend(fake_test)

 
    np.random.shuffle(combined_train_files)
    np.random.shuffle(combined_val_files)
    np.random.shuffle(combined_test_files)
    

    # Balance Train Set by downsampling the majority class
    train_reals = [f for f in combined_train_files if f[1] == 0]
    train_fakes = [f for f in combined_train_files if f[1] == 1]
    min_train = min(len(train_reals), len(train_fakes))
    train_list = train_reals[:min_train] + train_fakes[:min_train]
    np.random.shuffle(train_list)
    print(f"  - Train set balanced: {len(train_list)} total files ({min_train} real, {min_train} fake).")

    # Balance Validation Set by downsampling the majority class
    val_reals = [f for f in combined_val_files if f[1] == 0]
    val_fakes = [f for f in combined_val_files if f[1] == 1]
    min_val = min(len(val_reals), len(val_fakes))
    val_list = val_reals[:min_val] + val_fakes[:min_val]
    np.random.shuffle(val_list)
    print(f"  - Validation set balanced: {len(val_list)} total files ({min_val} real, {min_val} fake).")


    test_list = combined_test_files
    test_reals_count = len([f for f in test_list if f[1] == 0])
    test_fakes_count = len([f for f in test_list if f[1] == 1])
    print(f"  - Test set prepared (unbalanced): {len(test_list)} total files ({test_reals_count} real, {test_fakes_count} fake).")


    detector = MTCNN()
    
    # Process all three sets
    wanted = {s.strip() for s in args.splits.split(',') if s.strip()}
    for name, lst in (('train', train_list), ('val', val_list), ('test', test_list)):
        if name in wanted:
            process_and_save_files(lst, os.path.join(args.output_dir, name), detector, args)
        else:
            print(f"\nskipping split '{name}' ({len(lst)} clips) - not in --splits")

    print("\n--- All Steps Complete ---")
    print(f"Processed data is located in: {args.output_dir}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-process and split Polyglot data using JSON manifests.")
    
    parser.add_argument('--output_dir', type=str, default='polyglot_processed_all_unbalanced', 
                        help='Directory to save the processed and split .npz files.')
    parser.add_argument('--image_size', type=int, default=299, 
                        help='The size to resize all cropped faces to.')
    parser.add_argument('--max_faces', type=int, default=20, 
                        help='Maximum number of faces to extract from each video.')
    parser.add_argument('--frame_stride', type=int, default=10, 
                        help='Process every Nth frame to speed up extraction.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help="Root of the extracted PolyGlotFake release. Must contain "
                             "json_file/, real/ and fake/. Defaults to the historical "
                             "hard-coded value ('polyglot_lang') relative to the working "
                             "directory, so existing invocations are unaffected.")
    parser.add_argument('--splits', type=str, default='train,val,test',
                        help="Comma-separated splits to WRITE. The train/val/test partition is "
                             "always computed over the full file list, so restricting this does "
                             "not change which clip belongs to which split - it only skips the "
                             "writing. Use 'train,val' to retrain without materialising the "
                             "~8-10 GiB test split, which is not read by any training script.")
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N new clips this run, then stop (0 = no limit). '
                             'Re-running resumes: clips already written are skipped.')
    parser.add_argument('--min_free_gib', type=float, default=1.0,
                        help='Abort before writing if the output volume has less free space '
                             'than this. Guards against filling a 15 GiB Drive mid-run.')
    add_parallel_args(parser)
    parser.add_argument('--sample_rate', type=int, default=16000, 
                        help='Sample rate for audio extraction.')


    args = parser.parse_args()
    main(args)