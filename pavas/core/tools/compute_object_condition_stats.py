"""Compute offline mean/std (z-score stats) for mass_f & velocity_f.

Usage (generic roots):
  python -m core.tools.compute_object_condition_stats \
	  --mass-root /path/to/mass_train \
	  --velocity-root /path/to/velocity_train \
	  --out stats/object_cond_train.json --verbose

The directory layout is assumed:
  mass_root/<video_id>/mass_estimation.json
  velocity_root/<video_id>/<prompt_id>/speeds/instant_speeds.csv

We only accumulate:
  mass: object mass values for objects that appear in at least one frame (kg)
  velocity: velocity values from frames where `presence == True` (m/s)

Outputs JSON with structure:
{
  "mass": {"mean": float, "std": float, "count": int},
  "velocity": {"mean": float, "std": float, "count": int}
}

You can copy emitted config snippet into object_cond.mass_norm / vel_norm.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch

from core.data.object_conditions import (
    label_map_from_velocity_csv,
    parse_velocity_csv,
    load_mass_json,
)


def index_mass_json(mass_root: Path) -> dict[str, Path]:
	"""Recursively index mass_estimation.json files under mass_root.

	Supports layouts like:
	  mass_root/<video_id>/mass_estimation.json
	  mass_root/split_0_json/<video_id>/mass_estimation.json
	  mass_root/split_*/*/<video_id>/mass_estimation.json (one level deeper tolerated)

	We treat the immediate parent directory name of the json as the video_id.
	If duplicates appear (same video across different splits), keep the first.
	"""
	idx: dict[str, Path] = {}
	if not mass_root.exists():
		return idx
	try:
		for p in mass_root.rglob('mass_estimation.json'):
			if not p.is_file():
				continue
			vid = p.parent.name
			if vid not in idx:
				idx[vid] = p
	except Exception:
		pass
	return idx

def iter_velocity_videos(velocity_root: Path):
	if not velocity_root.exists():
		return
	try:
		for d in sorted(velocity_root.iterdir()):
			if d.is_dir():
				yield d.name
	except Exception:
		return


def compute_stats(mass_root: Path,
                  velocity_root: Path,
                  tv: Optional[int] = None,
                  max_videos: Optional[int] = None,
                  verbose: bool = False,
                  incremental_out: Optional[Path] = None,
                  debug_mass_samples: int = 0,
                  mass_max: float = 10000.0,
                  mass_min: float = 0.0,
                  log_mass_values: bool = False,
				  log_mass_limit: Optional[int] = None,
				  log_mass_file: Optional[Path] = None,
				  stats_interval: int = 100):
	mass_sum = 0.0
	mass_sq_sum = 0.0
	mass_n = 0
	mass_clipped = 0
	# log(mass) stats (log1p of per_prompt_mass after clipping)
	mass_log_sum = 0.0
	mass_log_sq_sum = 0.0
	vel_sum = 0.0
	vel_sq_sum = 0.0
	vel_n = 0
	# log(velocity) stats (log1p per scalar velocity after clipping)
	vel_log_sum = 0.0
	vel_log_sq_sum = 0.0
	vel_clipped = 0
	vel_outlier_warned = 0
	vel_missing_n = 0  # Number of frames with missing velocity because the object is absent.

	# Initialize the incremental stats file if it does not exist yet.
	if incremental_out is not None:
		try:
			incremental_out.parent.mkdir(parents=True, exist_ok=True)
			if not incremental_out.exists():
				incremental_out.write_text(json.dumps({'status': 'initializing'}, indent=2))
		except Exception:
			pass

	# Build mass index once (video_id -> json path)
	mass_index = index_mass_json(mass_root)
	missing_mass_files = 0
	debug_print_left = debug_mass_samples

	logged_mass_records = 0
	log_fh = None
	if log_mass_file is not None:
		try:
			log_mass_file.parent.mkdir(parents=True, exist_ok=True)
			log_fh = log_mass_file.open('w')
			log_fh.write('# video_id,prompt_id,label_ids(object ordering),mass_vector(presence-filtered),raw_mass_vector(full O),presence_any_mask(0/1 per object)\n')
		except Exception:
			log_fh = None
	for i, vid in enumerate(iter_velocity_videos(velocity_root)):
		if max_videos is not None and i >= max_videos:
			break
		vid_dir = velocity_root / vid
		try:
			prompt_dirs = [d for d in vid_dir.iterdir() if d.is_dir() and d.name.isdigit()]
		except Exception:
			continue
		if not prompt_dirs:
			continue
		for pdir in prompt_dirs:
			vel_csv = pdir / 'speeds' / 'instant_speeds.csv'
			if not vel_csv.exists():
				continue
			lid2obj = label_map_from_velocity_csv(vel_csv)
			if not lid2obj:
				continue
			label_ids_sorted = sorted(lid2obj.keys())
			try:
				vel, present_csv, _vel_missing = parse_velocity_csv(vel_csv, label_ids_sorted, tv=tv)
			except Exception:
				continue
			# Resolve mass json path via index (nested split directories supported)
			mass_json = mass_index.get(vid)
			if mass_json is None:
				missing_mass_files += 1
				mass_f = torch.zeros((len(label_ids_sorted), 1), dtype=torch.float32)
			else:
				try:
					mass_f = load_mass_json(mass_json, {lid: lid2obj[lid] for lid in label_ids_sorted})  # (O,1)
				except Exception:
					mass_f = torch.zeros((len(label_ids_sorted), 1), dtype=torch.float32)
				else:
					if debug_print_left > 0:
						try:
							weights_list = mass_f.squeeze(-1).tolist()
							print(f"[debug][mass] vid={vid} path={mass_json} weights={weights_list}")
						except Exception:
							pass
						debug_print_left -= 1

			presence_any = present_csv.any(dim=1)  # (O,) object present at least one frame
			if presence_any.any():
				m_valid = mass_f[presence_any]  # (K,1)
				if m_valid.numel() > 0:
					before = m_valid.clone()
					low = mass_min if mass_min is not None else 0.0
					high = mass_max if mass_max is not None else float('inf')
					m_valid.clamp_(min=low, max=high)
					mass_clipped += int((before != m_valid).sum().item())
					# Use one sample per `(video, prompt_dir)`: the clipped mean over present objects.
					per_prompt_mass = m_valid.mean().item()
					mass_sum += per_prompt_mass
					mass_sq_sum += per_prompt_mass * per_prompt_mass
					# Accumulate log1p(per_prompt_mass).
					m_log = math.log1p(per_prompt_mass)
					mass_log_sum += m_log
					mass_log_sq_sum += m_log * m_log
					mass_n += 1
					if log_mass_values and (log_mass_limit is None or logged_mass_records < log_mass_limit):
						try:
							mass_vec_full = mass_f.squeeze(-1).tolist()
							mass_vec_used = m_valid.squeeze(-1).tolist()
							presence_mask_list = presence_any.to(torch.int).tolist()
							record = {
								'video': vid,
								'prompt_dir': pdir.name,
								'label_ids': label_ids_sorted,
								'mass_full': mass_vec_full,
								'mass_present_values': mass_vec_used,
								'presence_any': presence_mask_list,
								'per_prompt_mass': per_prompt_mass
							}
							line = json.dumps(record, ensure_ascii=False)
							print(f"[mass-log] {line}")
							if log_fh is not None:
								log_fh.write(line + '\n')
							logged_mass_records += 1
						except Exception:
							pass

			pres_mask = present_csv.unsqueeze(-1)  # (O,T,1)
			vel_valid = vel[pres_mask]
			if vel_valid.numel() > 0:
				# Velocity clipping / outlier detection parameters (hard-coded defaults; can expose via CLI later)
				# We treat negative or NaN as 0 (defensive) before processing.
				vel_valid = vel_valid.float()
				vel_valid[~torch.isfinite(vel_valid)] = 0.0
				vel_valid.clamp_(min=0.0)  # ensure non-negative for log1p
				# Optional clipping if set globally via environment (simple hook) or attribute on function (future extension)
				vel_clip_max = getattr(compute_stats, '_vel_clip_max', None)
				vel_warn_threshold = getattr(compute_stats, '_vel_warn_threshold', None)
				if vel_clip_max is not None:
					before = vel_valid.clone()
					vel_valid.clamp_(max=float(vel_clip_max))
					vel_clipped += int((before != vel_valid).sum().item())
				if vel_warn_threshold is not None and vel_outlier_warned < 20:
					mask_warn = vel_valid > vel_warn_threshold
					if mask_warn.any():
						vals = vel_valid[mask_warn][:8].view(-1).tolist()
						print(f"[warn][vel-outlier] vid={vid} prompt_dir={pdir.name} sample={vals} threshold={vel_warn_threshold}")
						vel_outlier_warned += 1
				vel_sum += vel_valid.sum().item()
				vel_sq_sum += (vel_valid ** 2).sum().item()
				vel_n += vel_valid.numel()
				# log1p velocity accumulation
				v_log = torch.log1p(vel_valid)
				vel_log_sum += v_log.sum().item()
				vel_log_sq_sum += (v_log * v_log).sum().item()
			# Accumulate the number of frames with missing velocity.
			if _vel_missing is not None:
				try:
					vel_missing_n += int(_vel_missing.sum().item())
				except Exception:
					pass

		if verbose and stats_interval > 0 and (i + 1) % stats_interval == 0:
			# Compute running mean/std via E[x^2] - E[x]^2.
			if mass_n > 0:
				mass_mean_now = mass_sum / mass_n
				mass_var_now = max(mass_sq_sum / mass_n - mass_mean_now * mass_mean_now, 1e-12)
				mass_std_now = math.sqrt(mass_var_now)
				mass_log_mean_now = mass_log_sum / mass_n
				mass_log_var_now = max(mass_log_sq_sum / mass_n - mass_log_mean_now * mass_log_mean_now, 1e-12)
				mass_log_std_now = math.sqrt(mass_log_var_now)
			else:
				mass_mean_now = 0.0; mass_std_now = 1.0; mass_log_mean_now = 0.0; mass_log_std_now = 1.0
			if vel_n > 0:
				vel_mean_now = vel_sum / vel_n
				vel_var_now = max(vel_sq_sum / vel_n - vel_mean_now * vel_mean_now, 1e-12)
				vel_std_now = math.sqrt(vel_var_now)
				vel_log_mean_now = vel_log_sum / vel_n
				vel_log_var_now = max(vel_log_sq_sum / vel_n - vel_log_mean_now * vel_log_mean_now, 1e-12)
				vel_log_std_now = math.sqrt(vel_log_var_now)
			else:
				vel_mean_now = 0.0; vel_std_now = 1.0; vel_log_mean_now = 0.0; vel_log_std_now = 1.0
			total_vel_slots = vel_n + vel_missing_n
			miss_ratio = (vel_missing_n / total_vel_slots) if total_vel_slots > 0 else 0.0
			print(
				f"[stats] processed {i+1} videos "
				f"mass_n={mass_n} mass_mean={mass_mean_now:.4f} mass_std={mass_std_now:.4f} "
				f"mass_log_mean={mass_log_mean_now:.4f} mass_log_std={mass_log_std_now:.4f} "
				f"vel_n={vel_n} vel_mean={vel_mean_now:.4f} vel_std={vel_std_now:.4f} "
				f"vel_log_mean={vel_log_mean_now:.4f} vel_log_std={vel_log_std_now:.4f} "
				f"vel_missing={vel_missing_n} miss_ratio={miss_ratio:.3f} "
				f"missing_mass_files={missing_mass_files} mass_clipped={mass_clipped}")
			# Print a quick config snippet (not saved) for convenience
			print(f"[config-snippet] mass_norm: {{ mean: {mass_mean_now:.6f}, std: {mass_std_now:.6f} }} mass_log_norm: {{ mean: {mass_log_mean_now:.6f}, std: {mass_log_std_now:.6f} }} vel_norm: {{ mean: {vel_mean_now:.6f}, std: {vel_std_now:.6f} }} vel_log_norm: {{ mean: {vel_log_mean_now:.6f}, std: {vel_log_std_now:.6f} }}")
			# Save incremental stats by overwriting the same file and marking it as partial.
			if incremental_out is not None:
				try:
					inc_stats = {
						'mass': {
							'mean': mass_mean_now,
							'std': mass_std_now,
							'log_mean': mass_log_mean_now,
							'log_std': mass_log_std_now,
							'count': mass_n,
						},
						'velocity': {
							'mean': vel_mean_now,
							'std': vel_std_now,
							'log_mean': vel_log_mean_now,
							'log_std': vel_log_std_now,
							'count': vel_n,
							'missing_count': vel_missing_n,
							'missing_ratio': miss_ratio,
						},
						'partial': True,
						'videos_processed': i + 1,
						'mass_log_mean': mass_log_mean_now,
						'mass_log_std': mass_log_std_now,
						'velocity_log_mean': vel_log_mean_now,
						'velocity_log_std': vel_log_std_now,
					}
					incremental_out.write_text(json.dumps(inc_stats, indent=2))
				except Exception as e:
					print(f"[warn] incremental save failed: {e}")

	def finalize(s, ss, n):
		if n == 0:
			return 0.0, 1.0, 0
		mean = s / n
		var = max(ss / n - mean * mean, 1e-12)
		std = math.sqrt(var)
		return mean, std, n

	mass_mean, mass_std, _ = finalize(mass_sum, mass_sq_sum, mass_n)
	mass_log_mean, mass_log_std, _ = finalize(mass_log_sum, mass_log_sq_sum, mass_n)
	vel_mean, vel_std, _ = finalize(vel_sum, vel_sq_sum, vel_n)
	vel_log_mean, vel_log_std, _ = finalize(vel_log_sum, vel_log_sq_sum, vel_n)
	if log_fh is not None:
		try:
			log_fh.close()
		except Exception:
			pass
	return {
		'mass': {'mean': mass_mean, 'std': mass_std, 'log_mean': mass_log_mean, 'log_std': mass_log_std,
				 'count': mass_n, 'missing_file_count': missing_mass_files,
				 'indexed_files': len(mass_index), 'clipped_count': mass_clipped,
				 'mass_min': mass_min, 'mass_max': mass_max},
		'velocity': {'mean': vel_mean, 'std': vel_std, 'log_mean': vel_log_mean, 'log_std': vel_log_std,
					 'count': vel_n, 'missing_count': vel_missing_n,
					 'missing_ratio': (vel_missing_n / (vel_n + vel_missing_n)) if (vel_n + vel_missing_n) > 0 else 0.0,
					 'vel_clipped': vel_clipped},
		# Expose log-space statistics at the root level for convenience.
		'mass_log_mean': mass_log_mean,
		'mass_log_std': mass_log_std,
		'velocity_log_mean': vel_log_mean,
		'velocity_log_std': vel_log_std,
	}


def main():
	ap = argparse.ArgumentParser(description='Compute offline mean/std for mass_f & velocity_f (object conditions).')
	ap.add_argument('--mass-root', type=Path, required=True, help='Root dir with <video_id>/mass_estimation.json folders')
	ap.add_argument('--velocity-root', type=Path, required=True, help='Root dir with <video_id>/<prompt_id>/speeds/instant_speeds.csv')
	ap.add_argument('--tv', type=int, default=None, help='Optional fixed Tv length; normally leave None')
	ap.add_argument('--max-videos', type=int, default=None, help='Limit number of videos (debug)')
	ap.add_argument('--out', type=Path, default=None, help='Write JSON stats to this file if provided')
	ap.add_argument('--verbose', action='store_true')
	ap.add_argument('--debug-mass-samples', type=int, default=0, help='Print first N video mass vectors for debugging')
	ap.add_argument('--mass-max', type=float, default=10000.0, help='Clip mass values above this (default 10000)')
	ap.add_argument('--mass-min', type=float, default=0.0, help='Clip mass values below this (default 0)')
	ap.add_argument('--log-mass-values', action='store_true', help='Log per (video,prompt) mass vectors that are accumulated (after presence filter)')
	ap.add_argument('--log-mass-limit', type=int, default=None, help='Limit number of mass log records (None = all)')
	ap.add_argument('--log-mass-file', type=Path, default=None, help='Optional file to append JSONL mass logs')
	ap.add_argument('--stats-interval', type=int, default=100, help='Log & partial-save interval in #videos (default 100)')
	ap.add_argument('--vel-max', type=float, default=None, help='Optional: clip velocity values above this before stats')
	ap.add_argument('--vel-warn-threshold', type=float, default=200.0, help='Warn if any velocity exceeds this (before clipping). Set very high to suppress.')
	args = ap.parse_args()

	# Propagate velocity clip/warn thresholds via attributes (quick way without modifying many params)
	if args.vel_max is not None:
		setattr(compute_stats, '_vel_clip_max', args.vel_max)
	else:
		if hasattr(compute_stats, '_vel_clip_max'):
			delattr(compute_stats, '_vel_clip_max')
	if args.vel_warn_threshold is not None:
		setattr(compute_stats, '_vel_warn_threshold', args.vel_warn_threshold)

	stats = compute_stats(args.mass_root, args.velocity_root, tv=args.tv, max_videos=args.max_videos, verbose=args.verbose,
					  incremental_out=(args.out if args.out is not None else None),
					  debug_mass_samples=args.debug_mass_samples,
					  mass_max=args.mass_max, mass_min=args.mass_min,
					  log_mass_values=args.log_mass_values,
					  log_mass_limit=args.log_mass_limit,
					  log_mass_file=args.log_mass_file,
					  stats_interval=args.stats_interval)
	print(json.dumps(stats, indent=2))

	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(stats, indent=2))
		print(f"[stats] wrote {args.out}")

	print('\n# --- config snippet (object_cond) ---')
	print('norm: { enable: true, epsilon: 1e-6, clip: null }')
	print(f"mass_norm:      {{ mean: {stats['mass']['mean']:.6f}, std: {stats['mass']['std']:.6f} }}")
	print(f"mass_log_norm:  {{ mean: {stats['mass']['log_mean']:.6f}, std: {stats['mass']['log_std']:.6f} }}")
	print(f"vel_norm:       {{ mean: {stats['velocity']['mean']:.6f}, std: {stats['velocity']['std']:.6f} }}")
	print(f"vel_log_norm:   {{ mean: {stats['velocity']['log_mean']:.6f}, std: {stats['velocity']['log_std']:.6f} }}")


if __name__ == '__main__':
	main()
