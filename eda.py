"""Exploratory Data Analysis (EDA) for the Possession Challenge.

Streams tracking data once and analyzes match kinematics, public ground truth,
smart tagging predictive power, and baseline model failure modes.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Project imports
from games import GAME, GAMES, half_video_blocks, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags
from starter.load_homography import load as load_homo
import evaluation.possession_metrics as pm

# -----------------------------------------------------------------------------
# 1. STREAM TRACKING DATA & BUILD PER-FRAME ARRAYS
# -----------------------------------------------------------------------------

def extract_tracking_features(game=GAME):
    """Stream ATD raw tracking file once and build per-frame feature arrays."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    n_frames = source_frames("tactical", game)
    pitch_l, pitch_w = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)
    
    # Pre-allocate arrays
    ball_vis = np.zeros(n_frames, dtype=bool)
    ball_x = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y = np.full(n_frames, np.nan, dtype=np.float32)
    ball_s = np.full(n_frames, np.nan, dtype=np.float32)
    
    n_home = np.zeros(n_frames, dtype=np.int16)
    n_away = np.zeros(n_frames, dtype=np.int16)
    
    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    
    centroid_hx = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_hy = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_ax = np.full(n_frames, np.nan, dtype=np.float32)
    centroid_ay = np.full(n_frames, np.nan, dtype=np.float32)
    
    mean_speed_h = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed_a = np.full(n_frames, np.nan, dtype=np.float32)
    
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
            sum_hx, sum_hy, sum_hs = 0.0, 0.0, 0.0
            count_a, count_as = 0, 0
            sum_ax, sum_ay, sum_as = 0.0, 0.0, 0.0
            
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
                    sum_hy += py
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
                    sum_ay += py
                    if has_s:
                        sum_as += ps
                        count_as += 1
                    if b_vis:
                        dx = px - bx
                        dy = py - by
                        d2 = dx * dx + dy * dy
                        if d2 < min_sq_a:
                            min_sq_a = d2
            
            n_home[idx] = count_h
            n_away[idx] = count_a
            
            if count_h > 0:
                centroid_hx[idx] = sum_hx / count_h
                centroid_hy[idx] = sum_hy / count_h
            if count_hs > 0:
                mean_speed_h[idx] = sum_hs / count_hs
            
            if count_a > 0:
                centroid_ax[idx] = sum_ax / count_a
                centroid_ay[idx] = sum_ay / count_a
            if count_as > 0:
                mean_speed_a[idx] = sum_as / count_as
            
            if b_vis:
                if min_sq_h < float("inf"):
                    min_dist_h[idx] = np.sqrt(min_sq_h)
                if min_sq_a < float("inf"):
                    min_dist_a[idx] = np.sqrt(min_sq_a)
    
    elapsed = time.time() - t0
    return {
        "n_frames": n_frames,
        "elapsed": elapsed,
        "ball_vis": ball_vis,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "ball_s": ball_s,
        "n_home": n_home,
        "n_away": n_away,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "centroid_hx": centroid_hx,
        "centroid_hy": centroid_hy,
        "centroid_ax": centroid_ax,
        "centroid_ay": centroid_ay,
        "mean_speed_h": mean_speed_h,
        "mean_speed_a": mean_speed_a,
    }


def compute_gap_lengths(vis_array):
    """Compute run lengths of missing (False) values in a boolean array."""
    gaps = []
    n = len(vis_array)
    i = 0
    while i < n:
        if not vis_array[i]:
            start = i
            while i < n and not vis_array[i]:
                i += 1
            gaps.append(i - start)
        else:
            i += 1
    return np.array(gaps, dtype=int)


def extract_segments(labels):
    """Extract contiguous (label, start, end, length) segments."""
    n = len(labels)
    if n == 0:
        return []
    segments = []
    cur_label = labels[0]
    start = 0
    for i in range(1, n):
        if labels[i] != cur_label:
            segments.append((cur_label, start, i, i - start))
            cur_label = labels[i]
            start = i
    segments.append((cur_label, start, n, n - start))
    return segments


def run_eda():
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA) - POSSESSION CHALLENGE")
    print("=" * 80)
    
    # Extract tracking features
    print("\n[1/5] Streaming tracking data and extracting features...")
    data = extract_tracking_features()
    ball_vis = data["ball_vis"]
    ball_x = data["ball_x"]
    ball_y = data["ball_y"]
    ball_s = data["ball_s"]
    n_home = data["n_home"]
    n_away = data["n_away"]
    min_dist_h = data["min_dist_h"]
    min_dist_a = data["min_dist_a"]
    mean_speed_h = data["mean_speed_h"]
    mean_speed_a = data["mean_speed_a"]
    
    # Basic sanity checks
    assert len(ball_vis) == 199319, f"Expected 199319 frames, got {len(ball_vis)}"
    pub_start, pub_end = 11185, 18685  # half-open [11185, 18685) = 7500 frames
    assert (pub_end - pub_start) == 7500, "Public window slice must be 7500 frames"
    print(f"  Processed {len(ball_vis):,} frames in {data['elapsed']:.2f}s.")
    
    # Load Homography & Smart Tagging
    print("\n[2/5] Loading homography and smart tagging...")
    _, reproj_error, _ = load_homo(GAME)
    tags = load_tags(GAME)
    print(f"  Homography loaded ({len(reproj_error):,} rows).")
    print(f"  Smart Tagging loaded ({len(tags)} instances).")
    
    # Load Ground Truth
    gt_path = os.path.join(os.path.dirname(__file__), "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt_df = pd.read_csv(gt_path)
    gt_labels = gt_df["label"].values
    assert len(gt_labels) == 199319
    gt_pub = gt_labels[pub_start:pub_end]
    
    # Graded match blocks
    blocks = half_video_blocks(GAME)
    graded_mask = np.zeros(len(ball_vis), dtype=bool)
    for half, first, last in blocks:
        graded_mask[first : last + 1] = True
    n_graded = graded_mask.sum()
    assert n_graded == 158857, f"Expected 158857 graded frames, got {n_graded}"
    
    # -------------------------------------------------------------------------
    # PART 1: GRADED MATCH OVERVIEW
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: GRADED MATCH OVERVIEW (158,857 frames across Half 1 & 2)")
    print("=" * 80)
    
    pitch_l, pitch_w = GAMES[GAME]["pitch"]
    ball_vis_graded = ball_vis[graded_mask]
    ball_x_graded = ball_x[graded_mask]
    ball_y_graded = ball_y[graded_mask]
    
    vis_count = int(np.sum(ball_vis_graded))
    vis_pct = 100.0 * vis_count / n_graded
    
    # Out of bounds
    oob_graded = (
        (ball_x_graded < 0) | (ball_x_graded > pitch_l) |
        (ball_y_graded < 0) | (ball_y_graded > pitch_w)
    )
    oob_vis_count = int(np.sum(oob_graded & ball_vis_graded))
    oob_all_pct = 100.0 * oob_vis_count / n_graded
    oob_vis_pct = 100.0 * oob_vis_count / vis_count if vis_count else 0.0
    
    # Missing gaps inside each half
    gaps_h1 = compute_gap_lengths(ball_vis[blocks[0][1] : blocks[0][2] + 1])
    gaps_h2 = compute_gap_lengths(ball_vis[blocks[1][1] : blocks[1][2] + 1])
    gaps_graded = np.concatenate([gaps_h1, gaps_h2])
    gaps_sec = gaps_graded / 25.0
    
    # Player counts
    n_tot_graded = (n_home + n_away)[graded_mask]
    nh_graded = n_home[graded_mask]
    na_graded = n_away[graded_mask]
    
    print(f"\n1. Ball Visibility & Pitch Boundaries:")
    print(f"  - Total graded frames:          {n_graded:,} (H1: {blocks[0][2]-blocks[0][1]+1:,}, H2: {blocks[1][2]-blocks[1][1]+1:,})")
    print(f"  - Ball visible:                 {vis_count:,} frames ({vis_pct:.2f}%)")
    print(f"  - Ball missing / occluded:      {n_graded - vis_count:,} frames ({100.0 - vis_pct:.2f}%)")
    print(f"  - Ball out-of-bounds:           {oob_vis_count:,} frames ({oob_vis_pct:.2f}% of visible, {oob_all_pct:.2f}% of graded)")
    
    print(f"\n2. Missing-Ball Gap Durations (N = {len(gaps_graded):,} gap episodes in graded match):")
    pcts = [50, 75, 90, 95, 99]
    gap_pct_vals = np.percentile(gaps_sec, pcts)
    print(f"  {'Metric':<12} | {'Seconds':<10} | {'Frames':<10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for p, v in zip(pcts, gap_pct_vals):
        print(f"  p{p:<10} | {v:>8.2f} s | {int(v * 25):>8} f")
    print(f"  {'Max':<12} | {gaps_sec.max():>8.2f} s | {gaps_graded.max():>8} f")
    print(f"  {'Mean':<12} | {gaps_sec.mean():>8.2f} s | {gaps_graded.mean():>8.1f} f")
    
    print(f"\n3. Track Slot Visibility Distribution (Players):")
    print(f"  {'Team':<12} | {'Mean':<6} | {'Min':<4} | {'p25':<4} | {'Median':<6} | {'p75':<4} | {'Max':<4}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}")
    print(f"  {'Home (BAR)':<12} | {nh_graded.mean():>6.2f} | {nh_graded.min():>4} | {int(np.percentile(nh_graded, 25)):>4} | {int(np.median(nh_graded)):>6} | {int(np.percentile(nh_graded, 75)):>4} | {nh_graded.max():>4}")
    print(f"  {'Away (ATH)':<12} | {na_graded.mean():>6.2f} | {na_graded.min():>4} | {int(np.percentile(na_graded, 25)):>4} | {int(np.median(na_graded)):>6} | {int(np.percentile(na_graded, 75)):>4} | {na_graded.max():>4}")
    print(f"  {'Total':<12} | {n_tot_graded.mean():>6.2f} | {n_tot_graded.min():>4} | {int(np.percentile(n_tot_graded, 25)):>4} | {int(np.median(n_tot_graded)):>6} | {int(np.percentile(n_tot_graded, 75)):>4} | {n_tot_graded.max():>4}")
    print(f"  - Frames with exactly 22 tracked players: {np.sum(n_tot_graded == 22):,} ({100.0 * np.mean(n_tot_graded == 22):.1f}%)")
    print(f"  - Frames with >= 20 tracked players:      {np.sum(n_tot_graded >= 20):,} ({100.0 * np.mean(n_tot_graded >= 20):.1f}%)")
    print(f"  - Frames with < 15 tracked players:       {np.sum(n_tot_graded < 15):,} ({100.0 * np.mean(n_tot_graded < 15):.1f}%)")

    # -------------------------------------------------------------------------
    # PART 2: PUBLIC GROUND TRUTH DEEP DIVE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 2: PUBLIC GROUND TRUTH (11,185..18,684, 7,500 frames = 5.0 mins) DEEP DIVE")
    print("=" * 80)
    
    gt_segs = extract_segments(gt_pub)
    seg_df = pd.DataFrame(gt_segs, columns=["label", "start", "end", "len"])
    seg_df["duration_s"] = seg_df["len"] / 25.0
    
    print("\n1. Ground Truth Segment Statistics:")
    print(f"  {'Label':<6} | {'Segments':<9} | {'Total Frames':<13} | {'Total Time':<11} | {'Mean (s)':<9} | {'Min (s)':<8} | {'Median (s)':<11} | {'Max (s)':<8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*13}-+-{'-'*11}-+-{'-'*9}-+-{'-'*8}-+-{'-'*11}-+-{'-'*8}")
    for lbl in ["H", "A", "D"]:
        sub = seg_df[seg_df["label"] == lbl]
        cnt = len(sub)
        tot_f = sub["len"].sum()
        tot_s = sub["duration_s"].sum()
        mean_s = sub["duration_s"].mean() if cnt else 0
        min_s = sub["duration_s"].min() if cnt else 0
        med_s = sub["duration_s"].median() if cnt else 0
        max_s = sub["duration_s"].max() if cnt else 0
        print(f"  {lbl:<6} | {cnt:>9} | {tot_f:>13,} | {tot_s:>9.2f} s | {mean_s:>9.2f} | {min_s:>8.2f} | {med_s:>11.2f} | {max_s:>8.2f}")
    poss_sub = seg_df[seg_df["label"].isin(["H", "A"])]
    print(f"  {'H+A':<6} | {len(poss_sub):>9} | {poss_sub['len'].sum():>13,} | {poss_sub['duration_s'].sum():>9.2f} s | {poss_sub['duration_s'].mean():>9.2f} | {poss_sub['duration_s'].min():>8.2f} | {poss_sub['duration_s'].median():>11.2f} | {poss_sub['duration_s'].max():>8.2f}")
    
    # Cross-tab: GT vs Ball Visibility
    bvis_pub = ball_vis[pub_start:pub_end]
    ct_vis = pd.crosstab(gt_pub, bvis_pub, rownames=["GT Label"], colnames=["Ball Visible"])
    ct_vis.columns = ["Missing (False)", "Visible (True)"]
    ct_vis["Total"] = ct_vis.sum(axis=1)
    ct_vis["Vis %"] = 100.0 * ct_vis["Visible (True)"] / ct_vis["Total"]
    print("\n2. GT Label vs Ball Visibility:")
    print(ct_vis.to_string())
    
    # Missing ball breakdown: what % of missing ball is alive vs dead?
    missing_pub_mask = ~bvis_pub
    n_miss_pub = missing_pub_mask.sum()
    miss_by_gt = pd.Series(gt_pub[missing_pub_mask]).value_counts()
    print(f"\n3. Missing-Ball Breakdown in Public Window (Total missing = {n_miss_pub} frames):")
    for lbl in ["H", "A", "D"]:
        cnt = miss_by_gt.get(lbl, 0)
        pct = 100.0 * cnt / n_miss_pub
        print(f"  - GT {lbl}: {cnt:>5} frames ({pct:>5.1f}%)")
    alive_miss = miss_by_gt.get("H", 0) + miss_by_gt.get("A", 0)
    print(f"  ==> CRUCIAL INSIGHT: {alive_miss} of {n_miss_pub} missing-ball frames ({100.0 * alive_miss / n_miss_pub:.1f}%) are ALIVE play (H or A)!")
    print(f"      The baseline maps every missing ball to Dead Ball (D), directly fragmenting real possessions.")
    
    # Cross-tab: Ball Out-of-bounds when visible
    bx_pub = ball_x[pub_start:pub_end]
    by_pub = ball_y[pub_start:pub_end]
    oob_pub = (bx_pub < 0) | (bx_pub > pitch_l) | (by_pub < 0) | (by_pub > pitch_w)
    df_pub_vis = pd.DataFrame({
        "gt": gt_pub[bvis_pub],
        "oob": oob_pub[bvis_pub],
    })
    ct_oob = pd.crosstab(df_pub_vis["gt"], df_pub_vis["oob"], rownames=["GT Label"], colnames=["Ball Out-Of-Bounds"])
    ct_oob.columns = ["In-Bounds", "Out-Of-Bounds"]
    ct_oob["Total"] = ct_oob.sum(axis=1)
    ct_oob["OOB %"] = 100.0 * ct_oob["Out-Of-Bounds"] / ct_oob["Total"]
    print("\n4. Ball Out-of-Bounds (when visible) vs GT Label:")
    print(ct_oob.to_string())
    
    # Cross-tab: Nearest Player Team & Proximity
    md_h_pub = min_dist_h[pub_start:pub_end]
    md_a_pub = min_dist_a[pub_start:pub_end]
    min_dist_pub = np.minimum(md_h_pub, md_a_pub)
    
    nearest_team = np.full(len(gt_pub), "None", dtype=object)
    nearest_team[bvis_pub & (md_h_pub < md_a_pub)] = "H_closest"
    nearest_team[bvis_pub & (md_a_pub < md_h_pub)] = "A_closest"
    nearest_team[bvis_pub & (md_h_pub == md_a_pub)] = "Tie"
    
    ct_near = pd.crosstab(gt_pub[bvis_pub], nearest_team[bvis_pub], rownames=["GT Label"], colnames=["Nearest Player Team"])
    print("\n5. Nearest Player Team vs GT Label (when ball is visible):")
    print(ct_near.to_string())
    
    # Proximity within 2 metres
    within_2m = bvis_pub & (min_dist_pub < 2.0)
    print(f"\n6. Proximity within 2.0m:")
    print(f"  - Visible frames with a player within 2.0m: {within_2m.sum():,} of {bvis_pub.sum():,} ({100.0 * within_2m.sum() / bvis_pub.sum():.1f}%)")
    ct_2m = pd.crosstab(gt_pub[within_2m], nearest_team[within_2m], rownames=["GT Label"], colnames=["Nearest Player Team (<2m)"])
    print(ct_2m.to_string())
    if "H_closest" in ct_2m.columns and "H" in ct_2m.index:
        h_acc = 100.0 * ct_2m.loc["H", "H_closest"] / ct_2m.loc["H"].sum()
        print(f"  - When player < 2m and GT is H: Home is closest {h_acc:.1f}% of the time.")
    if "A_closest" in ct_2m.columns and "A" in ct_2m.index:
        a_acc = 100.0 * ct_2m.loc["A", "A_closest"] / ct_2m.loc["A"].sum()
        print(f"  - When player < 2m and GT is A: Away is closest {a_acc:.1f}% of the time.")
        
    # Kinematics & Homography Reprojection Error
    rep_pub = reproj_error[pub_start:pub_end]
    sh_pub = mean_speed_h[pub_start:pub_end]
    sa_pub = mean_speed_a[pub_start:pub_end]
    
    kin_df = pd.DataFrame({
        "gt": gt_pub,
        "mean_speed_h": sh_pub,
        "mean_speed_a": sa_pub,
        "reproj_error": rep_pub,
    })
    print("\n7. Kinematics and Homography Error across GT Classes (Mean +/- Std):")
    print(f"  {'GT':<4} | {'Mean Speed H (m/s)':<20} | {'Mean Speed A (m/s)':<20} | {'Reproj Error (px)':<20}")
    print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    for lbl in ["H", "A", "D"]:
        sub = kin_df[kin_df["gt"] == lbl]
        sh_m, sh_s = sub["mean_speed_h"].mean(), sub["mean_speed_h"].std()
        sa_m, sa_s = sub["mean_speed_a"].mean(), sub["mean_speed_a"].std()
        re_m, re_s = sub["reproj_error"].mean(), sub["reproj_error"].std()
        print(f"  {lbl:<4} | {sh_m:>6.2f} +/- {sh_s:>5.2f}       | {sa_m:>6.2f} +/- {sa_s:>5.2f}       | {re_m:>6.2f} +/- {re_s:>5.2f}")

    # -------------------------------------------------------------------------
    # PART 3: SMART TAGGING ALIGNMENT & PREDICTIVE POWER
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: SMART TAGGING ALIGNMENT & PREDICTIVE POWER")
    print("=" * 80)
    
    # Frame coverage
    tagged_graded = np.zeros(len(ball_vis), dtype=bool)
    tagged_pub = np.zeros(len(ball_vis), dtype=bool)
    for t in tags:
        s, e = t["start_frame"], t["end_frame"]
        tagged_graded[s:e] = True
    
    cov_match = 100.0 * np.sum(tagged_graded[graded_mask]) / n_graded
    cov_pub = 100.0 * np.sum(tagged_graded[pub_start:pub_end]) / 7500
    print(f"\n1. Smart Tagging Frame Coverage:")
    print(f"  - Total instances in XML:        {len(tags)}")
    print(f"  - Graded frames covered (>=1 tag): {np.sum(tagged_graded[graded_mask]):,} / {n_graded:,} ({cov_match:.1f}%)")
    print(f"  - Public window covered:         {np.sum(tagged_graded[pub_start:pub_end]):,} / 7,500 ({cov_pub:.1f}%)")
    
    # Tag team confusion matrix in public window
    # We analyze both: (A) Core offensive phases and (B) All active tags
    CORE_OFFENSIVE = {'BUILD UP', 'PROGRESSION', 'ATTACKING TRANSITION', 'CREATING CHANCES', 'LONG BALL', 'GOALS'}
    
    tag_team_core = []
    for f in range(pub_start, pub_end):
        active = [t for t in tags if t["start_frame"] <= f < t["end_frame"] and t["code"] in CORE_OFFENSIVE]
        teams = set(t["tags"].get("Team") for t in active if t["tags"].get("Team"))
        if "Barcelona" in teams and "Atletic Club" in teams:
            tag_team_core.append("Both")
        elif "Barcelona" in teams:
            tag_team_core.append("Barcelona (H)")
        elif "Atletic Club" in teams:
            tag_team_core.append("Atletic Club (A)")
        elif "N/A" in teams:
            tag_team_core.append("N/A")
        else:
            tag_team_core.append("No active tag")
            
    ct_tag = pd.crosstab(pd.Series(tag_team_core, name="SmartTag Core Phase"), pd.Series(gt_pub, name="GT Label"))
    print("\n2. Confusion Matrix: Smart Tagging Core Offensive Phase Team vs GT Label:")
    print(ct_tag.to_string())
    
    # Calculate precision
    if "Barcelona (H)" in ct_tag.index:
        bar_tot = ct_tag.loc["Barcelona (H)"].sum()
        bar_h = ct_tag.loc["Barcelona (H)", "H"] if "H" in ct_tag.columns else 0
        print(f"  - When SmartTag is 'Barcelona': {bar_h} / {bar_tot} frames are GT H (Precision = {100.0 * bar_h / bar_tot:.1f}%)")
    if "Atletic Club (A)" in ct_tag.index:
        ath_tot = ct_tag.loc["Atletic Club (A)"].sum()
        ath_a = ct_tag.loc["Atletic Club (A)", "A"] if "A" in ct_tag.columns else 0
        print(f"  - When SmartTag is 'Atletic Club': {ath_a} / {ath_tot} frames are GT A (Precision = {100.0 * ath_a / ath_tot:.1f}%)")
        
    # Set Pieces alignment with D intervals
    print("\n3. Set Pieces Alignment with Dead-Ball (D) Intervals:")
    pub_sp = [t for t in tags if t["code"] == "SET PIECES" and not (t["end_frame"] <= pub_start or t["start_frame"] >= pub_end)]
    d_segs = seg_df[seg_df["label"] == "D"]
    
    print(f"  GT Dead-Ball (D) Segments in Public Window (N = {len(d_segs)}):")
    for idx, row in d_segs.iterrows():
        abs_s = pub_start + row["start"]
        abs_e = pub_start + row["end"]
        print(f"    D Segment {idx}: frames {abs_s}..{abs_e} ({row['len']} f, {row['duration_s']:.2f} s)")
    
    print(f"\n  Set Piece Events Overlapping Public Window (N = {len(pub_sp)}):")
    for sp in pub_sp:
        t_type = sp["tags"].get("Type", "Unknown")
        t_team = sp["tags"].get("Team", "N/A")
        dur_s = (sp["end_frame"] - sp["start_frame"]) / 25.0
        print(f"    Type: {t_type:<20} | Team: {t_team:<15} | Frames: {sp['start_frame']}..{sp['end_frame']} ({dur_s:.2f} s)")
    print("  Key Takeaway: Set Pieces mark the restart of play; the restart instant coincides with the D -> H/A transition.")

    # -------------------------------------------------------------------------
    # PART 4: BASELINE FAILURE MODE DIAGNOSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 4: BASELINE FAILURE MODE DIAGNOSIS")
    print("=" * 80)
    
    res_dir = os.path.join(os.path.dirname(__file__), "evaluation", "results")
    variants = ["baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"]
    gt_codes = pm.label_codes(gt_labels)
    
    print("\n1. Segmentation & Metrics Comparison on Public Window (GT: 17 possessions, 4 D segments):")
    print(f"  {'Submission':<20} | {'macroF1':<7} | {'spellF1':<7} | {'Alive/Dead':<10} | {'Pred Spells':<11} | {'Pred D Segs':<11} | {'Mean Spell (s)':<14}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*11}-+-{'-'*11}-+-{'-'*14}")
    
    baseline_preds = {}
    for name in variants:
        path = os.path.join(res_dir, f"{name}.csv")
        pred_codes = pm.label_codes(pm.read_label_csv(path))
        baseline_preds[name] = pred_codes
        
        # Public window metrics
        ev = pm.evaluate(gt_codes, pred_codes)
        
        # Segment counts on public window
        pred_pub_labels = [pm.LABELS[c] for c in pred_codes[pub_start:pub_end]]
        pred_pub_segs = extract_segments(pred_pub_labels)
        p_spells = [s for s in pred_pub_segs if s[0] in ("H", "A")]
        p_d = [s for s in pred_pub_segs if s[0] == "D"]
        mean_dur = np.mean([s[3] for s in p_spells]) / 25.0 if p_spells else 0.0
        
        print(f"  {name:<20} | {ev['macro_f1']:>7.4f} | {ev['spell_f1_mean']:>7.4f} | {ev['alive_dead_f1']:>10.4f} | {len(p_spells):>11} | {len(p_d):>11} | {mean_dur:>14.2f}")
    
    print(f"  {'GT Truth':<20} | {'1.0000':>7} | {'1.0000':>7} | {'1.0000':>10} | {17:>11} | {4:>11} | {14.19:>14.2f}")
    
    # Graded match segment count comparison
    print("\n2. Predicted Segment Counts on Full Graded Match (158,857 frames):")
    print(f"  {'Submission':<20} | {'Total Segments':<14} | {'H Spells':<9} | {'A Spells':<9} | {'D Segments':<10} | {'Mean Spell (s)':<14}")
    print(f"  {'-'*20}-+-{'-'*14}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*14}")
    for name in variants:
        pred_codes = baseline_preds[name]
        # Evaluate on graded frames
        # Extract segments across graded blocks
        h1_segs = extract_segments([pm.LABELS[c] for c in pred_codes[blocks[0][1] : blocks[0][2] + 1]])
        h2_segs = extract_segments([pm.LABELS[c] for c in pred_codes[blocks[1][1] : blocks[1][2] + 1]])
        all_segs = h1_segs + h2_segs
        h_cnt = sum(1 for s in all_segs if s[0] == "H")
        a_cnt = sum(1 for s in all_segs if s[0] == "A")
        d_cnt = sum(1 for s in all_segs if s[0] == "D")
        spells = [s for s in all_segs if s[0] in ("H", "A")]
        mean_dur = np.mean([s[3] for s in spells]) / 25.0 if spells else 0.0
        print(f"  {name:<20} | {len(all_segs):>14,} | {h_cnt:>9,} | {a_cnt:>9,} | {d_cnt:>10,} | {mean_dur:>14.2f}")
        
    # In-Memory Diagnostic: What happens if D is NOT triggered by short ball-missing dropouts?
    print("\n3. In-Memory Diagnostic: Mitigating Ball Dropout Fragmentation on baseline_smooth50:")
    base_codes = baseline_preds["baseline_smooth50"]
    ev_base = pm.evaluate(gt_codes, base_codes)
    
    print(f"  Baseline (baseline_smooth50): macroF1 = {ev_base['macro_f1']:.4f}, spellF1 = {ev_base['spell_f1_mean']:.4f}, pred spells = {ev_base['n_pred_spells']}")
    print(f"\n  Testing post-processing bridging of short D segments (dropout suppression):")
    print(f"  {'Max Gap Bridged':<16} | {'macroF1':<7} | {'d_macro':<8} | {'spellF1':<7} | {'d_spell':<8} | {'Pred Spells':<11}")
    print(f"  {'-'*16}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*11}")
    
    for gap_s in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        gap_f = int(gap_s * 25)
        mod_codes = base_codes.copy()
        
        # Bridge D runs <= gap_f
        n = len(mod_codes)
        i = 0
        while i < n:
            if mod_codes[i] == pm.D:
                start = i
                while i < n and mod_codes[i] == pm.D:
                    i += 1
                length = i - start
                if length <= gap_f:
                    left = mod_codes[start - 1] if start > 0 else pm.D
                    right = mod_codes[i] if i < n else pm.D
                    if left == right and left in (pm.H, pm.A):
                        mod_codes[start:i] = left
                    elif left in (pm.H, pm.A):
                        mod_codes[start:i] = left
            else:
                i += 1
                
        res = pm.evaluate(gt_codes, mod_codes)
        d_macro = res["macro_f1"] - ev_base["macro_f1"]
        d_spell = res["spell_f1_mean"] - ev_base["spell_f1_mean"]
        print(f"  <= {gap_s:>4.1f}s ({gap_f:>3} f)   | {res['macro_f1']:>7.4f} | {d_macro:>+8.4f} | {res['spell_f1_mean']:>7.4f} | {d_spell:>+8.4f} | {res['n_pred_spells']:>11}")
        
    print("\n" + "=" * 80)
    print("EDA COMPLETE.")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
