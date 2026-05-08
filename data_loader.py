# ============================================================
#  data_loader.py
#  Loads all Cricsheet IPL JSON files into a clean DataFrame
# ============================================================

import os
import json
import pandas as pd
import numpy as np


def load_ipl_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads all .json files from the given Cricsheet folder.
    Returns:
        ball_df   — one row per delivery (both innings)
        match_df  — one row per match (metadata)
    """
    ball_rows  = []
    match_rows = []

    files = [f for f in os.listdir(path) if f.endswith(".json")]
    print(f"  Found {len(files)} JSON files in {path}")

    for file in files:
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)

        info   = data.get('info', {})
        teams  = info.get('teams', [None, None])
        team1, team2 = teams[0], teams[1]

        outcome = info.get('outcome', {})
        winner  = outcome.get('winner')          # None if no result
        win_by  = outcome.get('by', {})          # {'runs': N} or {'wickets': N}

        players_info = info.get('players', {})
        team_captains = {}
        for tn in [team1, team2]:
            if tn and tn in players_info:
                plist = players_info[tn]
                team_captains[tn] = plist[0] if plist else None

        stage = info.get('stage', '') or ''
        is_playoff = any(k in stage.lower() for k in ['qualifier', 'eliminator', 'final', 'semi'])

        # Skip no-result matches (rain, abandoned)
        if winner is None:
            continue

        # ── Match-level metadata ──────────────────────────────
        match_rows.append({
            'match_id'    : file,
            'team1'       : team1,
            'team2'       : team2,
            'winner'      : winner,
            'win_by_runs' : win_by.get('runs'),
            'win_by_wkts' : win_by.get('wickets'),
            'venue'       : info.get('venue'),
            'city'        : info.get('city'),
            'season'      : info.get('season'),
            'date'        : info.get('dates', [None])[0],
            'toss_winner' : info.get('toss', {}).get('winner'),
            'toss_decision': info.get('toss', {}).get('decision'),
            'stage'       : stage,
            'is_playoff'  : is_playoff,
            'captain_team1': team_captains.get(team1),
            'captain_team2': team_captains.get(team2),
        })

        # ── Ball-level data ───────────────────────────────────
        for inn_idx, inning in enumerate(data.get('innings', [])):
            inn_num = inn_idx + 1
            if inn_num > 2:          # skip super overs
                continue

            bat_team = inning['team']
            bowl_team = team2 if bat_team == team1 else team1
            legal_ball = 0           # count only legal deliveries

            for over in inning.get('overs', []):
                ov_num = over['over']          # 0-indexed in Cricsheet

                for delivery in over.get('deliveries', []):
                    extras     = delivery.get('extras', {})
                    extra_type = list(extras.keys())[0] if extras else None

                    is_wide   = extra_type == 'wides'
                    is_noball = extra_type == 'noballs'
                    is_legal  = not (is_wide or is_noball)

                    if is_legal:
                        legal_ball += 1

                    wickets = delivery.get('wickets', [])
                    wkt_kind = wickets[0]['kind'] if wickets else None
                    dismissed_player = wickets[0].get('player_out') if wickets else None
                    wkt_fielder = None
                    if wickets:
                        fielders = wickets[0].get('fielders', [])
                        if fielders:
                            wkt_fielder = fielders[0].get('name')

                    ball_rows.append({
                        'match_id'          : file,
                        'inning'            : inn_num,
                        'batting_team'      : bat_team,
                        'bowling_team'      : bowl_team,
                        'over'              : ov_num + 1,          # 1-indexed
                        'legal_ball_in_inn' : legal_ball,
                        'is_legal'          : int(is_legal),
                        'batter'            : delivery.get('batter'),
                        'bowler'            : delivery.get('bowler'),
                        'non_striker'       : delivery.get('non_striker'),
                        'runs_batter'       : delivery['runs']['batter'],
                        'runs_extras'       : delivery['runs']['extras'],
                        'runs_total'        : delivery['runs']['total'],
                        'extra_type'        : extra_type or 'none',
                        'is_wicket'         : 1 if wickets else 0,
                        'dismissal_kind'    : wkt_kind,
                        'dismissed_player'  : dismissed_player,
                        'fielder'           : wkt_fielder,
                        'team1'             : team1,
                        'team2'             : team2,
                        'winner'            : winner,
                        'season'            : info.get('season'),
                    })

    ball_df  = pd.DataFrame(ball_rows)
    match_df = pd.DataFrame(match_rows)

    print(f"  Loaded {len(match_df)} matches, {len(ball_df):,} deliveries")
    return ball_df, match_df


def build_innings2(ball_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the raw ball DataFrame, build a clean 2nd-innings
    ball-by-ball dataset with cumulative game-state features.
    All features are pure arithmetic — no model involved.
    """

    # ── 1st innings totals → target for 2nd innings ──────────
    inn1 = ball_df[ball_df['inning'] == 1]
    targets = (
        inn1.groupby('match_id')['runs_total'].sum()
        .reset_index()
        .rename(columns={'runs_total': 'inn1_total'})
    )
    targets['target'] = targets['inn1_total'] + 1   # need 1 more to win

    # ── Keep only 2nd innings ─────────────────────────────────
    inn2 = ball_df[ball_df['inning'] == 2].copy()
    inn2 = inn2.merge(targets[['match_id', 'target']], on='match_id')
    inn2 = inn2.sort_values(['match_id', 'legal_ball_in_inn']).reset_index(drop=True)

    # ── Cumulative features ───────────────────────────────────
    inn2['runs_scored']     = inn2.groupby('match_id')['runs_total'].cumsum()
    inn2['wickets_fallen']  = inn2.groupby('match_id')['is_wicket'].cumsum()
    inn2['wickets_left']    = 10 - inn2['wickets_fallen']
    inn2['balls_bowled']    = inn2['legal_ball_in_inn']     # alias for clarity
    inn2['balls_left']      = (120 - inn2['legal_ball_in_inn']).clip(lower=0)
    inn2['runs_left']       = (inn2['target'] - inn2['runs_scored']).clip(lower=0)

    # ── Run rates ─────────────────────────────────────────────
    inn2['crr'] = inn2['runs_scored'] / (inn2['balls_bowled'] / 6).replace(0, np.nan)
    inn2['rrr'] = np.where(
        inn2['balls_left'] > 0,
        inn2['runs_left'] / (inn2['balls_left'] / 6),
        0
    )
    inn2['pressure'] = inn2['rrr'] - inn2['crr']

    # ── Match phase ───────────────────────────────────────────
    inn2['match_phase'] = pd.cut(
        inn2['over'],
        bins=[0, 6, 15, 20],
        labels=['Powerplay', 'Middle', 'Death']
    ).astype(str)

    # ── Result: did batting team win? ─────────────────────────
    inn2['result'] = (inn2['batting_team'] == inn2['winner']).astype(int)

    # ── Resources remaining (simple proxy) ───────────────────
    inn2['resources_left'] = (inn2['balls_left'] / 120) * (inn2['wickets_left'] / 10)

    # ── Clean ─────────────────────────────────────────────────
    inn2.replace([np.inf, -np.inf], np.nan, inplace=True)
    inn2 = inn2.dropna(subset=['runs_left', 'balls_left', 'wickets_left'])
    inn2 = inn2[inn2['balls_left'] >= 0]

    print(f"  2nd innings dataset: {len(inn2):,} balls, {inn2['match_id'].nunique()} matches")
    return inn2


def load_and_build(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load JSON → build innings2 → return all three.
    """
    ball_df, match_df = load_ipl_data(path)
    inn2_df = build_innings2(ball_df)
    return ball_df, match_df, inn2_df
