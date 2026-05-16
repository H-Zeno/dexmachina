"""WandbVideoObserver — rl_games AlgoObserver that periodically uploads
training-rollout videos to the active wandb run.

How it works:
- ``base_env._render_headless`` already captures frames on every training
  step whenever ``base_env._recording`` is True (constructors enable that
  path via ``--record_video``).
- This observer runs a small FSM driven by ``after_print_stats``:
    * idle: wait until ``epoch_num % video_interval == 0`` and flip
      ``base_env._recording = True`` so frames start accumulating.
    * recording: once ``len(_recorded_frames) >= max_video_frames``,
      export the mp4 to ``<log_dir>/videos/<exp>_epoch<NNN>.mp4``,
      ``wandb.log({"video": wandb.Video(path), "video/epoch": epoch})``,
      and reset to idle.

The companion ``IsaacAlgoObserver`` continues to handle scalar logging
unchanged — we subclass it so the existing reward / episode metrics
keep flowing.
"""

import os
import time

import wandb

from rl_games.common.algo_observer import IsaacAlgoObserver


class WandbVideoObserver(IsaacAlgoObserver):
    def __init__(self, log_root, exp_name, video_interval=200):
        super().__init__()
        self._video_root = os.path.join(log_root, exp_name, "videos")
        os.makedirs(self._video_root, exist_ok=True)
        self._exp_name = exp_name
        self._video_interval = int(video_interval)
        self._base_env = None
        self._last_recorded_epoch = -1

    def after_init(self, algo):
        super().after_init(algo)
        # vec_env -> rl_games wrapper -> dexmachina rl_games_wrapper -> BaseEnv
        env = algo.vec_env.env
        while hasattr(env, "env") and env.__class__.__name__ != "BaseEnv":
            env = env.env
        self._base_env = env
        if not getattr(self._base_env, "record_video", False):
            print("[WandbVideoObserver] base_env.record_video is False; video upload disabled.")
            self._base_env = None

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        if self._base_env is None or wandb.run is None:
            return

        recording = bool(getattr(self._base_env, "_recording", False))
        frames = getattr(self._base_env, "_recorded_frames", [])
        max_frames = int(getattr(self._base_env, "max_video_frames", 0))

        if not recording:
            if epoch_num > 0 and epoch_num % self._video_interval == 0 and epoch_num != self._last_recorded_epoch:
                self._base_env.start_recording()
                self._last_recorded_epoch = epoch_num
                print(f"[WandbVideoObserver] start_recording at epoch {epoch_num} (target {max_frames} frames)")
            return

        if len(frames) < max_frames:
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self._video_root, f"epoch{epoch_num:06d}_{ts}.mp4")
        self._base_env.export_video(path)

        log_payload: dict = {
            "video/epoch": epoch_num,
            "video/frame": frame,
        }
        if os.path.exists(path):
            log_payload["video"] = wandb.Video(path, format="mp4")
            print(f"[WandbVideoObserver] uploaded {path} at epoch {epoch_num}")
        else:
            print(f"[WandbVideoObserver] export_video did not produce {path}; skipping primary video upload")

        # Auxiliary cameras (e.g. 'grid' macro view of ~100 envs). Each gets
        # its own wandb panel under video/<cam_name> so the dashboard can
        # show the front close-up and the parallel-env overview side by side.
        aux_paths = self._base_env.export_aux_videos(path) if hasattr(self._base_env, "export_aux_videos") else {}
        for cam_name, aux_path in aux_paths.items():
            if os.path.exists(aux_path):
                log_payload[f"video/{cam_name}"] = wandb.Video(aux_path, format="mp4")
                print(f"[WandbVideoObserver] uploaded {aux_path} at epoch {epoch_num}")
            else:
                print(f"[WandbVideoObserver] export_aux_videos did not produce {aux_path}; skipping")

        if len(log_payload) > 2:  # has at least one video besides the bookkeeping epoch/frame keys
            wandb.log(log_payload)
