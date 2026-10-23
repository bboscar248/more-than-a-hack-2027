"""Physics-informed HMM with Viterbi decoding v0.3 for football possession inference.

Decodes possession states (0=H, 1=A, 2=D) by fusing:
  1. Optical Tracking Proximity with Physical In-Bounds & Interaction Gating:
     - Possession evidence (score_h / score_a) is strictly gated to in-bounds balls
       (not is_oob) within physical interaction range (min(dh, da) < 2.2m). Out-of-bounds
       balls (e.g. retrieved behind the goal line) cannot be possessed by any team.
  2. Dead-Ball Ball-Placement Filter & False Stoppage Island Prevention:
     - Distinguishes dead-ball setup/placement (lone player next to slow ball while others walk)
       from active match play: suppresses false dead-ball exit and dampens proximity.
     - Merges spurious dead-ball placement islands (D -> H/A -> D where mean player speed < 1.65 m/s
       or length < 40 frames) back into continuous stoppages.
  3. Contact-Gated & Physical Restart Viterbi Decoding:
     - Dead-ball stoppage state (D) can only transition to live play (H or A) upon genuine
       physical restart: ball accelerated into play (kalman_bs > 3.0 m/s or ball_s > 3.0 m/s)
       or collective match sprint (roll_speed > 2.20 m/s).
  4. Kinematic Trajectory & 2D Kalman Smoother (`kalman_smooth_ball`):
     - Continuous ball velocity estimation across optical tracking dropouts (<= 125 frames = 5.0s).
  5. Discrete Control Events & Pass Bridging (`build_pass_bridges`):
     - Uncontested control touches (d < 1.8m, opponent > 2.2m) bridge flight intervals (<= 150 frames = 6.0s).
     - Same-team passes receive pass-bridge support (+2.0 log-evidence); turnovers credit the passer
       until the opponent's first control touch (+1.5 log-evidence).
  6. Collective Team Advancement Velocity Drift:
     - Self-calibrated team pitch advancement velocity gently resolves team momentum during
       live aerial occlusions (not b_vis and roll_speed > 1.85 m/s).
  7. Conditional Pitch Boundary Bridging:
     - Genuine out-of-bounds stoppages (>= 5 OOB frames) are preserved down to 40 frames (1.6s)
       to retain rapid throw-ins, while non-OOB false micro-stoppages are bridged up to 90 frames (3.6s).

Broadcast ATD Feasibility Note:
  Analysis of BAR-ATH_broadcast_atd.txt (164,488 frames vs 199,319 tactical frames) revealed 122
  temporal discontinuities (cuts, close-up replays, non-linear timeline jumps). Without broadcast-to-tactical
  homography alignment timecodes, direct timeline warping carries severe timeline drift risk. v0.3
  therefore relies on tactical ATD kinematics enhanced with discrete control-event bridging and Kalman smoothing.

Outputs frame,label predictions across all 199,319 frames, with ungraded frames labeled 'X'.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Project modules
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags
import evaluation.possession_metrics as pm

# Constants
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
GT_CSV = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# State indices
STATE_H, STATE_A, STATE_D = 0, 1, 2
STATE_LABELS = np.array(["H", "A", "D"], dtype=object)

# Smart Tagging taxonomy
OFFENSIVE_CODES = {
    "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION",
    "CREATING CHANCES", "LONG BALL", "GOALS",
}
DEFENSIVE_CODES = {
    "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION",
}


def extract_features(game=GAME):
    """Stream tracking data once and extract per-frame physical kinematics and centroids."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch_l, pitch_w = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)

    ball_vis = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)

    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)

    mean_speed_h = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed_a = np.full(n_frames, np.nan, dtype=np.float32)

    centroid_hx = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_ax = np.full(n_frames, np.nan, dtype=np.float32)

    t0 = time.time()
    with open(raw_path, "r") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            idx = int(parts[0]) - 1
            if idx < 0 or idx >= n_frames:
                continue

            ball = parts[2]
            b_vis = ball[0] != "N"
            bx, by = 0.0, 0.0
            if b_vis:
                ball_vis[idx] = True
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                bx = float(ball[:c1]) * pitch_l
                by = float(ball[c1 + 1 : c2]) * pitch_w
                ball_x[idx] = bx
                ball_y[idx] = by
                bs_str = ball[c2 + 1 :].strip()
                if bs_str[0] != "N":
                    ball_s[idx] = float(bs_str)

            count_h, count_hs = 0, 0
            sum_hs = 0.0
            sum_hx = 0.0
            count_a, count_as = 0, 0
            sum_as = 0.0
            sum_ax = 0.0

            min_sq_h = float("inf")
            min_sq_a = float("inf")

            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px = float(group[:c1]) * pitch_l
                py = float(group[c1 + 1 : c2]) * pitch_w
                ps_str = group[c2 + 1 :].strip()
                has_s = ps_str[0] != "N"
                ps = float(ps_str) if has_s else 0.0

                slot_team = team_of_track[slot]
                if slot_team == 0:  # Home
                    count_h += 1
                    sum_hx += px
                    if has_s:
                        sum_hs += ps
                        count_hs += 1
                    if b_vis:
                        dx = px - bx
                        dy = py - by
                        d2 = dx * dx + dy * dy
                        if d2 < min_sq_h:
                            min_sq_h = d2
                else:  # Away
                    count_a += 1
                    sum_ax += px
                    if has_s:
                        sum_as += ps
                        count_as += 1
                    if b_vis:
                        dx = px - bx
                        dy = py - by
                        d2 = dx * dx + dy * dy
                        if d2 < min_sq_a:
                            min_sq_a = d2

            if count_hs > 0:
                mean_speed_h[idx] = sum_hs / count_hs
            if count_as > 0:
                mean_speed_a[idx] = sum_as / count_as
            if count_h > 0:
                centroid_hx[idx] = sum_hx / count_h
            if count_a > 0:
                centroid_ax[idx] = sum_ax / count_a

            if b_vis:
                if min_sq_h < float("inf"):
                    min_dist_h[idx] = np.sqrt(min_sq_h)
                if min_sq_a < float("inf"):
                    min_dist_a[idx] = np.sqrt(min_sq_a)

    player_speed = 0.5 * (mean_speed_h + mean_speed_a)
    oob = (ball_x < 0) | (ball_x > pitch_l) | (ball_y < 0) | (ball_y > pitch_w)

    return {
        "n_frames": n_frames,
        "elapsed": time.time() - t0,
        "ball_vis": ball_vis,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_s": ball_s,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "player_speed": player_speed,
        "oob": oob,
        "centroid_hx": centroid_hx,
        "centroid_ax": centroid_ax,
    }


def kalman_smooth_ball(bx, by, bvis, max_gap=125, dt=0.04):
    """2D Constant-Velocity Kalman/state-space smoother for ball trajectory."""
    n = len(bx)
    smooth_x = bx.copy()
    smooth_y = by.copy()
    smooth_vis = bvis.copy()

    # Step 1: Linear state-space bridging across short dropouts
    i = 0
    while i < n:
        if not smooth_vis[i]:
            start = i
            while i < n and not smooth_vis[i]:
                i += 1
            gap = i - start
            if gap <= max_gap and start > 0 and i < n:
                x0, y0 = smooth_x[start - 1], smooth_y[start - 1]
                x1, y1 = smooth_x[i], smooth_y[i]
                if not np.isnan(x0) and not np.isnan(x1):
                    alpha = np.linspace(0, 1, gap + 2)[1:-1]
                    smooth_x[start:i] = x0 + alpha * (x1 - x0)
                    smooth_y[start:i] = y0 + alpha * (y1 - y0)
                    smooth_vis[start:i] = True
        else:
            i += 1

    # Step 2: Smoothing filter to remove tracking jitter
    s_x = pd.Series(smooth_x).interpolate(limit=5).bfill().ffill()
    s_y = pd.Series(smooth_y).interpolate(limit=5).bfill().ffill()
    smooth_x = s_x.rolling(5, center=True, min_periods=1).mean().values
    smooth_y = s_y.rolling(5, center=True, min_periods=1).mean().values

    # Step 3: Compute continuous kinematic velocity
    dx = np.gradient(smooth_x) / dt
    dy = np.gradient(smooth_y) / dt
    kalman_bs = np.sqrt(dx * dx + dy * dy)
    kalman_bs[~smooth_vis] = np.nan

    return smooth_x, smooth_y, kalman_bs, smooth_vis


def build_pass_bridges(b_vis, dh, da, max_flight_gap=150):
    """Identify discrete control touches and bridge flight intervals between passes."""
    n = len(b_vis)
    bridge_h = np.zeros(n, dtype=np.float32)
    bridge_a = np.zeros(n, dtype=np.float32)

    touches = []
    for f in range(n):
        if b_vis[f]:
            h_ctrl = dh[f] < 1.8 and (np.isnan(da[f]) or da[f] > 2.2)
            a_ctrl = da[f] < 1.8 and (np.isnan(dh[f]) or dh[f] > 2.2)
            if h_ctrl and not a_ctrl:
                touches.append((f, 0))  # 0 = Home
            elif a_ctrl and not h_ctrl:
                touches.append((f, 1))  # 1 = Away

    for k in range(len(touches) - 1):
        t1, team1 = touches[k]
        t2, team2 = touches[k + 1]
        gap = t2 - t1
        if 1 < gap <= max_flight_gap:
            if team1 == team2:
                if team1 == 0:
                    bridge_h[t1:t2] += 2.0
                else:
                    bridge_a[t1:t2] += 2.0
            else:
                if team1 == 0:
                    bridge_h[t1:t2] += 1.5
                else:
                    bridge_a[t1:t2] += 1.5

    return bridge_h, bridge_a


def compute_smart_tag_priors(n_frames, roll_speed, features, kalman_bs, game=GAME):
    """Convert SportsCode Smart Tagging XML into per-frame log-prior potentials."""
    tags = load_tags(game)
    prior_h = np.zeros(n_frames, dtype=np.float32)
    prior_a = np.zeros(n_frames, dtype=np.float32)
    prior_d = np.zeros(n_frames, dtype=np.float32)

    ball_vis = features["ball_vis"]
    ball_s = features["ball_s"]
    oob = features["oob"]

    for t in tags:
        s, e = t["start_frame"], t["end_frame"]
        code = t["code"]
        team = t["tags"].get("Team")

        if code in OFFENSIVE_CODES:
            if team == "Barcelona":
                prior_h[s:e] += 1.0
            elif team == "Atletic Club":
                prior_a[s:e] += 1.0
        elif code in DEFENSIVE_CODES:
            if team == "Barcelona":
                prior_a[s:e] += 0.8
            elif team == "Atletic Club":
                prior_h[s:e] += 0.8
        elif code == "SET PIECES":
            for f in range(s, min(e, n_frames)):
                rps = roll_speed[f]
                b_v = ball_vis[f]
                bs = ball_s[f]
                k_bs = kalman_bs[f]
                is_oob = oob[f]

                # Ball kicked into play check using Kalman kinematic velocity
                if (not is_oob) and (not np.isnan(k_bs) and k_bs > 4.0):
                    is_setup = False
                else:
                    is_setup = (
                        (rps < 1.70)
                        or is_oob
                        or (not b_v)
                        or (b_v and not np.isnan(bs) and bs < 1.0)
                    )

                if is_setup:
                    prior_d[f] += 2.0
                else:
                    if team == "Barcelona":
                        prior_h[f] += 1.2
                    elif team == "Atletic Club":
                        prior_a[f] += 1.2

    return prior_h, prior_a, prior_d


def viterbi_decode_restart_gated(emissions, base_trans, can_restart):
    """Vectorized Viterbi path decoding with physical restart gating on dead ball."""
    T, K = emissions.shape
    dp = np.zeros((T, K), dtype=np.float64)
    backtrack = np.zeros((T, K), dtype=np.int32)

    dp[0] = emissions[0]
    for t in range(1, T):
        dyn_trans = base_trans.copy()
        # Physical Restart Barrier: Stoppage (D) cannot exit to live play without restart event
        if not can_restart[t]:
            dyn_trans[STATE_D, STATE_H] -= 25.0
            dyn_trans[STATE_D, STATE_A] -= 25.0

        scores = dp[t - 1][:, None] + dyn_trans + emissions[t][None, :]
        backtrack[t] = np.argmax(scores, axis=0)
        dp[t] = np.max(scores, axis=0)

    best_path = np.zeros(T, dtype=np.int32)
    best_path[-1] = np.argmax(dp[-1])
    for t in range(T - 2, -1, -1):
        best_path[t] = backtrack[t + 1, best_path[t + 1]]

    return best_path


def compute_half_emissions(
    first, last, features, priors, pass_bridges, kalman_bs, roll_speed, adv_vel_h
):
    """Compute per-frame log-emissions with in-bounds physical gating and placement filtering."""
    T = last - first + 1
    emissions = np.zeros((T, 3), dtype=np.float64)

    ball_vis = features["ball_vis"]
    min_dist_h = features["min_dist_h"]
    min_dist_a = features["min_dist_a"]
    oob = features["oob"]
    ball_s = features["ball_s"]

    prior_h, prior_a, prior_d = priors
    pass_h, pass_a = pass_bridges

    for i in range(T):
        f = first + i
        b_v = ball_vis[f]
        dh = min_dist_h[f]
        da = min_dist_a[f]
        rps = roll_speed[f]
        is_oob = oob[f]
        bs = ball_s[f]
        k_bs = kalman_bs[f]

        # -------------------------------------------------------------
        # State 2: Dead-Ball (D) Emission
        # -------------------------------------------------------------
        score_d = 0.0
        if is_oob:
            score_d += 5.0

        # Player speed stoppage evidence
        if rps < 1.40:
            score_d += 6.0 * (1.40 - rps) / 0.40
        elif rps < 1.75:
            score_d += 3.0 * (1.75 - rps) / 0.35
        elif rps > 2.10:
            score_d -= 3.0 * (rps - 2.10) / 0.50

        # Missing ball combined with player kinematics
        if not b_v:
            if rps < 1.80:
                score_d += 2.2
            elif rps > 2.10:
                score_d -= 2.0

        # Dead-ball placement condition: lone player standing next to slow ball while others walk
        is_placement = (
            (rps <= 1.65)
            and (np.isnan(k_bs) or k_bs < 2.5)
            and (not np.isnan(dh) and not np.isnan(da) and min(dh, da) < 2.0 and max(dh, da) > 5.0)
        )

        # Active in-bounds play protection:
        if b_v and not is_oob and not np.isnan(dh) and not np.isnan(da):
            min_d_val = min(dh, da)
            if min_d_val < 2.0:
                if not is_placement:
                    score_d -= 4.0

        # Stationary ball with no close opposing duel
        if b_v and not np.isnan(bs) and bs < 0.5:
            if not np.isnan(dh) and not np.isnan(da) and min(dh, da) > 2.0:
                score_d += 1.5

        score_d += prior_d[f] * 1.5
        emissions[i, STATE_D] = score_d

        # -------------------------------------------------------------
        # States 0 & 1: Possession (H vs A) Emissions
        # -------------------------------------------------------------
        score_h = 0.0
        score_a = 0.0

        # Physical In-Bounds Possession Gating:
        # A team can only possess the ball if the ball is inside the pitch boundaries
        if b_v and not is_oob and not np.isnan(dh) and not np.isnan(da):
            min_d_val = min(dh, da)
            if min_d_val < 2.2:
                is_duel = (dh < 1.8) and (da < 1.8)
                if dh < 1.3:
                    h_add = 3.5 if not is_duel else 1.2
                elif dh < 2.2:
                    h_add = 2.0 if not is_duel else 0.8
                else:
                    h_add = 0.0

                if da < 1.3:
                    a_add = 3.5 if not is_duel else 1.2
                elif da < 2.2:
                    a_add = 2.0 if not is_duel else 0.8
                else:
                    a_add = 0.0

                diff = np.clip(da - dh, -4.0, 4.0)

                # Dampen proximity evidence during dead-ball placement
                if is_placement:
                    h_add *= 0.25
                    a_add *= 0.25
                    diff *= 0.25

                score_h += h_add + 0.5 * diff
                score_a += a_add - 0.5 * diff

        # Pass bridge log-evidence (aerial flights between touches)
        score_h += pass_h[f]
        score_a += pass_a[f]

        # Smart tagging tactical priors
        score_h += prior_h[f] * 1.0
        score_a += prior_a[f] * 1.0

        # Team advancement velocity drift during live occlusions
        if not b_v and rps > 1.85:
            drift = np.clip(adv_vel_h[i] * 0.15, -0.4, 0.4)
            score_h += drift
            score_a -= drift

        emissions[i, STATE_H] = score_h
        emissions[i, STATE_A] = score_a

    return emissions


def extract_possession_spells(labels):
    """Extract consecutive (label, start, end, len) runs."""
    n = len(labels)
    if n == 0:
        return []
    runs = []
    cur = labels[0]
    start = 0
    for i in range(1, n):
        if labels[i] != cur:
            runs.append((cur, start, i, i - start))
            cur = labels[i]
            start = i
    runs.append((cur, start, n, n - start))
    return runs


def clean_spurious_segments_v3(
    labels, oob_half, rps_half, min_d_frames=90, min_d_oob_frames=40, min_deflection_frames=35
):
    """Eliminate physically implausible micro-stoppages, placement islands, and isolated deflections.

    1. Conditional Dead-Ball Bridging:
       - Preserves genuine pitch boundary exits down to 40 frames (1.6s).
       - Bridges false non-OOB micro-stoppages up to 90 frames (3.6s).
    2. Dead-Ball Placement Island filter:
       - Merges false possession flickers bounded by stoppages (D -> H/A -> D) back into D
         when collective player speed < 1.65 m/s or segment length < 40 frames (1.6s).
    3. Deflection Bridging:
       - Bridges turnover flickers < 35 frames (1.4s) between same-team possessions.
    """
    cleaned = labels.copy()
    n = len(cleaned)

    # Pass 1: Conditional Dead-Ball Bridging
    if min_d_frames > 0:
        i = 0
        while i < n:
            if cleaned[i] == "D":
                start = i
                while i < n and cleaned[i] == "D":
                    i += 1
                length = i - start

                is_genuine_oob = np.sum(oob_half[start:i]) >= 5
                thresh = min_d_oob_frames if is_genuine_oob else min_d_frames

                if length < thresh:
                    left = cleaned[start - 1] if start > 0 else "D"
                    right = cleaned[i] if i < n else "D"
                    if left in ("H", "A") and right in ("H", "A"):
                        cleaned[start:i] = left
                    elif left in ("H", "A"):
                        cleaned[start:i] = left
                    elif right in ("H", "A"):
                        cleaned[start:i] = right
            else:
                i += 1

    # Pass 2: Dead-Ball Placement Island Filter (D -> H/A -> D)
    i = 0
    while i < n:
        if cleaned[i] in ("H", "A"):
            team = cleaned[i]
            start = i
            while i < n and cleaned[i] == team:
                i += 1
            length = i - start
            left = cleaned[start - 1] if start > 0 else None
            right = cleaned[i] if i < n else None
            if left == "D" and right == "D":
                mean_spd = np.mean(rps_half[start:i])
                if mean_spd < 1.65 or length < 40:
                    cleaned[start:i] = "D"
        else:
            i += 1

    # Pass 3: Micro-deflection bridging between same team
    if min_deflection_frames > 0:
        i = 0
        while i < n:
            if cleaned[i] in ("H", "A"):
                cur_team = cleaned[i]
                start = i
                while i < n and cleaned[i] == cur_team:
                    i += 1
                length = i - start
                left = cleaned[start - 1] if start > 0 else None
                right = cleaned[i] if i < n else None
                other_team = "A" if cur_team == "H" else "H"
                if length < min_deflection_frames and left == other_team and right == other_team:
                    cleaned[start:i] = other_team
            else:
                i += 1

    return cleaned


def build_and_evaluate(
    p_switch=70.0,
    p_enter_d=55.0,
    p_exit_d=15.0,
    min_d_frames=90,
    min_d_oob_frames=40,
    min_deflection_frames=35,
    game=GAME,
    out_csv=os.path.join(RESULTS_DIR, "model_v03.csv"),
):
    """Build full submission, run sanity checks, and evaluate performance."""
    print("=" * 80)
    print("VITERBI HMM POSSESSION MODEL v0.3 (model_v03.py)")
    print("=" * 80)

    # 1. Physical feature extraction
    print("\n[1/6] Extracting physical signals and centroids from tracking data...")
    features = extract_features(game)
    n_frames = features["n_frames"]
    print(f"  Processed {n_frames:,} frames in {features['elapsed']:.2f}s.")

    # Rolling player speed (window 50 frames = 2.0s)
    roll_speed_50 = (
        pd.Series(features["player_speed"])
        .rolling(50, min_periods=1, center=True)
        .mean()
        .values
    )

    # 2. Kinematic Trajectory Smoothing & Velocity Computation
    print("\n[2/6] Running 2D Kalman state-space smoother on ball trajectory...")
    t_kalman_start = time.time()
    _, _, kalman_bs, _ = kalman_smooth_ball(
        features["ball_x"], features["ball_y"], features["ball_vis"]
    )
    print(f"  Computed kinematic ball velocity in {time.time() - t_kalman_start:.2f}s.")

    # 3. Discrete Control Events & Pass Bridging
    print("\n[3/6] Building discrete control events and pass bridges...")
    t_bridge_start = time.time()
    pass_bridges = build_pass_bridges(
        features["ball_vis"], features["min_dist_h"], features["min_dist_a"]
    )
    print(f"  Built pass bridges in {time.time() - t_bridge_start:.2f}s.")

    # 4. Smart Tagging Priors
    print("\n[4/6] Computing tactical phase priors from Smart Tagging...")
    priors = compute_smart_tag_priors(n_frames, roll_speed_50, features, kalman_bs, game)

    # 5. Transition Matrix & Half-by-Half Decoding
    print("\n[5/6] Running contact & restart-gated Viterbi decoding...")
    base_trans = np.zeros((3, 3), dtype=np.float64)
    base_trans[0, 1] = -p_switch
    base_trans[1, 0] = -p_switch
    base_trans[0, 2] = -p_enter_d
    base_trans[1, 2] = -p_enter_d
    base_trans[2, 0] = -p_exit_d
    base_trans[2, 1] = -p_exit_d

    # Physical restart condition
    can_restart = (
        (~np.isnan(kalman_bs) & (kalman_bs > 3.0))
        | (~np.isnan(features["ball_s"]) & (features["ball_s"] > 3.0))
        | (roll_speed_50 > 2.20)
    )

    blocks = half_video_blocks(game)
    full_labels = np.full(n_frames, "X", dtype=object)

    t_decode_start = time.time()
    for half, first, last in blocks:
        T = last - first + 1
        c_res_half = can_restart[first : last + 1]
        oob_half = features["oob"][first : last + 1]
        rps_half = roll_speed_50[first : last + 1]

        # Self-calibrated team pitch advancement velocity
        chx = features["centroid_hx"][first : last + 1]
        cax = features["centroid_ax"][first : last + 1]
        att_dir_h = 1.0 if np.nanmean(chx) < np.nanmean(cax) else -1.0
        c_mid = 0.5 * (
            pd.Series(chx).interpolate().bfill().ffill()
            + pd.Series(cax).interpolate().bfill().ffill()
        )
        c_mid_smooth = c_mid.rolling(50, center=True, min_periods=1).mean().values
        adv_vel_h = att_dir_h * np.gradient(c_mid_smooth) * 25.0

        emissions = compute_half_emissions(
            first, last, features, priors, pass_bridges, kalman_bs, roll_speed_50, adv_vel_h
        )

        path = viterbi_decode_restart_gated(emissions, base_trans, c_res_half)
        raw_half_labels = STATE_LABELS[path]

        cleaned_half_labels = clean_spurious_segments_v3(
            raw_half_labels,
            oob_half,
            rps_half,
            min_d_frames,
            min_d_oob_frames,
            min_deflection_frames,
        )
        full_labels[first : last + 1] = cleaned_half_labels
        print(f"  Half {half} ({last - first + 1:,} frames) decoded.")
    print(f"  Total decoding time: {time.time() - t_decode_start:.2f}s.")

    # 6. Rigorous Sanity Checks
    print("\n[6/6] Validating submission integrity & football realism...")
    assert len(full_labels) == n_frames, f"Length {len(full_labels)} != {n_frames}"
    assert set(np.unique(full_labels)).issubset({"H", "A", "D", "X"}), "Invalid labels found"

    # Ungraded frames check: must be strictly X
    graded_mask = np.zeros(n_frames, dtype=bool)
    for half, first, last in blocks:
        graded_mask[first : last + 1] = True
    assert (full_labels[~graded_mask] == "X").all(), "Ungraded frames must be labeled X"
    assert not (full_labels[graded_mask] == "X").any(), "Graded frames must NOT be X"

    # Possession segment statistics across full match
    h1_runs = extract_possession_spells(full_labels[blocks[0][1] : blocks[0][2] + 1])
    h2_runs = extract_possession_spells(full_labels[blocks[1][1] : blocks[1][2] + 1])
    all_runs = h1_runs + h2_runs
    spells = [r for r in all_runs if r[0] in ("H", "A")]
    d_runs = [r for r in all_runs if r[0] == "D"]
    durations = [r[3] / 25.0 for r in spells]

    n_spells_full = len(spells)
    mean_dur_full = float(np.mean(durations))

    print(f"  Total predicted possessions: {n_spells_full}")
    print(f"  Total dead-ball segments:     {len(d_runs)}")
    print(f"  Mean possession duration:     {mean_dur_full:.2f} s (median: {np.median(durations):.2f} s)")

    assert 160 <= n_spells_full <= 340, (
        f"Possession count {n_spells_full} outside expected range [160, 340]"
    )
    assert 9.0 <= mean_dur_full <= 18.0, (
        f"Mean spell duration {mean_dur_full:.2f}s outside realistic range [9.0, 18.0]"
    )
    print("  Sanity checks passed successfully!")

    # Write submission CSV
    folder = os.path.dirname(out_csv)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)

    df_sub = pd.DataFrame({"frame": np.arange(n_frames), "label": full_labels})
    df_sub.to_csv(out_csv, index=False)
    print(f"\nSaved submission to: {out_csv}")

    # Evaluate on public ground truth
    gt_df = pd.read_csv(GT_CSV)
    gt_codes = pm.label_codes(gt_df["label"].values)
    pred_codes = pm.label_codes(full_labels)
    eval_res = pm.evaluate(gt_codes, pred_codes)

    print("\n" + "=" * 80)
    print("EVALUATION ON PUBLIC GROUND TRUTH (5.0 MIN WINDOW)")
    print("=" * 80)
    print(f"  macroF1:              {eval_res['macro_f1']:.4f}")
    print(f"  spellF1 (headline):   {eval_res['spell_f1_mean']:.4f}")
    print(f"  spellF1 @ 0.5 IoU:    {eval_res['spell_f1_at_headline']:.4f}")
    print(f"  Alive/Dead F1:        {eval_res['alive_dead_f1']:.4f}")
    print(f"  Predicted Spells:     {eval_res['n_pred_spells']} (GT: 17)")
    print(f"  Mean Spell Duration:  {eval_res['mean_spell_duration_s']:.2f} s (GT: {eval_res['gt_mean_spell_duration_s']:.2f} s)")
    print(f"  Per-Class F1:         H: {eval_res['per_class']['H']['f1']:.4f}  |  A: {eval_res['per_class']['A']['f1']:.4f}  |  D: {eval_res['per_class']['D']['f1']:.4f}")

    # Side-by-side segment inspection
    pub_start, pub_end = 11185, 18685
    pred_pub = full_labels[pub_start:pub_end]

    runs = pm.valid_runs(gt_codes[pub_start:pub_end])
    pred_codes_pub = pm.label_codes(pred_pub)
    gt_spells = pm.possession_spells(gt_codes[pub_start:pub_end], runs)
    pred_spells = pm.possession_spells(pred_codes_pub, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)

    matched_gt = {}
    matched_pred = set()
    for iou, gi, pi in overlaps:
        if iou < 0.1:
            continue
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt[gi] = (pi, iou)
        matched_pred.add(pi)

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE GT POSSESSION SPELLS VS PREDICTIONS (IoU)")
    print("=" * 80)
    print(f"{'GT#':<4} {'Team':<5} {'GT Start':<8} {'GT End':<8} {'GT Dur(s)':<10} {'Matched Pred (Start..End)':<26} {'Pred Dur(s)':<12} {'IoU':<6}")
    for i, (gs, ge, gc) in enumerate(gt_spells):
        gt_team = pm.LABELS[gc]
        if i in matched_gt:
            pi, iou = matched_gt[i]
            ps, pe, pc = pred_spells[pi]
            p_str = f"{pub_start + ps}..{pub_start + pe} ({pm.LABELS[pc]})"
            p_dur = f"{(pe - ps) / 25.0:.2f}s"
            print(f"{i:<4} {gt_team:<5} {pub_start + gs:<8} {pub_start + ge:<8} {(ge - gs) / 25.0:>7.2f}s   {p_str:<26} {p_dur:<12} {iou:>6.3f}")
        else:
            cands = [(iou, pi) for (iou, gi, pi) in overlaps if gi == i]
            if cands:
                best_iou, pi = max(cands)
                ps, pe, pc = pred_spells[pi]
                p_str = f"[unmatched] {pub_start + ps}..{pub_start + pe} ({pm.LABELS[pc]})"
                p_dur = f"{(pe - ps) / 25.0:.2f}s"
                print(f"{i:<4} {gt_team:<5} {pub_start + gs:<8} {pub_start + ge:<8} {(ge - gs) / 25.0:>7.2f}s   {p_str:<26} {p_dur:<12} {best_iou:>6.3f}")
            else:
                print(f"{i:<4} {gt_team:<5} {pub_start + gs:<8} {pub_start + ge:<8} {(ge - gs) / 25.0:>7.2f}s   {'--- NONE (MISSED) ---':<26} {'---':<12} {0.0:>6.3f}")

    return eval_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viterbi Possession Model v0.3")
    parser.add_argument("--p-switch", type=float, default=70.0, help="Penalty for H<->A switch")
    parser.add_argument("--p-enter-d", type=float, default=55.0, help="Penalty for entering D")
    parser.add_argument("--p-exit-d", type=float, default=15.0, help="Penalty for exiting D")
    parser.add_argument("--min-d", type=int, default=90, help="Minimum dead-ball frames (3.6s)")
    parser.add_argument("--min-d-oob", type=int, default=40, help="Minimum dead-ball frames for OOB events (1.6s)")
    parser.add_argument("--min-def", type=int, default=35, help="Minimum deflection frames (1.4s)")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "model_v03.csv"))
    args = parser.parse_args()

    build_and_evaluate(
        p_switch=args.p_switch,
        p_enter_d=args.p_enter_d,
        p_exit_d=args.p_exit_d,
        min_d_frames=args.min_d,
        min_d_oob_frames=args.min_d_oob,
        min_deflection_frames=args.min_def,
        out_csv=args.out,
    )
