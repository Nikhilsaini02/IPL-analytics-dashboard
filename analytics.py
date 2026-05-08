# ============================================================
#  analytics.py
#  All analysis functions — pure aggregations, no ML
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
#  WIN PROBABILITY — PURE HISTORICAL LOOKUP
# ============================================================

def build_win_prob_lookup(inn2: pd.DataFrame) -> pd.DataFrame:
    """
    For every (balls_left_bin, runs_left_bin, wickets_left) combo,
    compute: win% = wins / total matches in that state.
    This is the ONLY win probability calculation — no model.
    """
    df = inn2.copy()

    # Bin balls_left into 6-ball (1-over) windows
    df['balls_bin'] = (df['balls_left'] // 6) * 6

    # Bin runs_left — tighter bins at low values
    df['runs_bin'] = pd.cut(
        df['runs_left'],
        bins=[0, 6, 12, 20, 30, 40, 55, 75, 100, 130, 175, 9999],
        labels=[3, 9, 16, 25, 35, 47, 65, 87, 115, 152, 250]
    ).astype(float)

    lookup = (
        df.groupby(['balls_bin', 'runs_bin', 'wickets_left'])
        .agg(
            total   = ('result', 'count'),
            wins    = ('result', 'sum')
        )
        .reset_index()
    )
    lookup['win_pct'] = (lookup['wins'] / lookup['total']).round(4)
    lookup['reliable'] = lookup['total'] >= 5    # flag sparse bins

    return lookup, df


def attach_win_prob(inn2: pd.DataFrame, lookup: pd.DataFrame, binned: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the historical win% back onto every ball.
    """
    df = binned.merge(
        lookup[['balls_bin', 'runs_bin', 'wickets_left', 'win_pct', 'total', 'reliable']],
        on=['balls_bin', 'runs_bin', 'wickets_left'],
        how='left'
    )
    # win_prob column = historical win_pct, nothing else
    df['win_prob'] = df['win_pct']
    df = df.sort_values(['match_id', 'legal_ball_in_inn']).reset_index(drop=True)

    # Ball-to-ball win prob change
    df['win_prob_prev']  = df.groupby('match_id')['win_prob'].shift(1)
    df['win_prob_delta'] = df['win_prob'] - df['win_prob_prev']
    df['swing']          = df['win_prob_delta'].abs()

    return df


# ============================================================
#  ANALYSIS MODULES
# ============================================================

def phase_analysis(df: pd.DataFrame) -> dict:
    """
    Over-by-over and phase-level win probability & swing stats.
    """
    by_over = (
        df.groupby('over')
        .agg(
            avg_win_prob = ('win_prob', 'mean'),
            avg_swing    = ('swing', 'mean'),
            total_balls  = ('win_prob', 'count'),
        )
        .reset_index()
    )

    by_phase = (
        df.groupby('match_phase')
        .agg(
            avg_win_prob  = ('win_prob', 'mean'),
            avg_swing     = ('swing', 'mean'),
            avg_pressure  = ('pressure', 'mean'),
            total_balls   = ('win_prob', 'count'),
        )
        .reset_index()
    )

    return {'by_over': by_over, 'by_phase': by_phase}


def team_analysis(df: pd.DataFrame) -> dict:
    """
    Chase win rates, volatility, and pressure profiles per team.
    """
    # One row per match for win rate
    match_level = (
        df.groupby(['match_id', 'batting_team'])
        .agg(result=('result', 'first'))
        .reset_index()
    )
    win_rates = (
        match_level.groupby('batting_team')
        .agg(
            matches  = ('result', 'count'),
            wins     = ('result', 'sum'),
        )
        .reset_index()
    )
    win_rates['win_rate'] = win_rates['wins'] / win_rates['matches']

    # Ball-level aggregations
    ball_level = (
        df.groupby('batting_team')
        .agg(
            avg_win_prob    = ('win_prob', 'mean'),
            volatility      = ('swing', 'mean'),       # avg |Δprob| per ball
            avg_pressure    = ('pressure', 'mean'),
        )
        .reset_index()
    )

    team_stats = win_rates.merge(ball_level, on='batting_team')
    team_stats = team_stats[team_stats['matches'] >= 10]   # filter tiny samples
    team_stats = team_stats.sort_values('win_rate', ascending=False)

    # Over-by-over journey per team
    journey = (
        df.groupby(['batting_team', 'over'])['win_prob']
        .mean()
        .reset_index()
        .rename(columns={'win_prob': 'avg_win_prob'})
    )

    return {'stats': team_stats, 'journey': journey}


def wicket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each wicket number (1st, 2nd, ... 10th),
    compute the average win prob before and after it falls.
    """
    wkt_balls = df[df['is_wicket'] == 1].copy()
    wkt_balls['wicket_number'] = wkt_balls['wickets_fallen'].astype(int)

    impact = (
        wkt_balls.groupby('wicket_number')
        .agg(
            avg_prob_before = ('win_prob_prev', 'mean'),
            avg_prob_after  = ('win_prob', 'mean'),
            count           = ('win_prob', 'count'),
        )
        .reset_index()
    )
    impact['prob_drop'] = (impact['avg_prob_before'] - impact['avg_prob_after']).round(4)
    impact = impact[impact['wicket_number'].between(1, 10)]
    return impact


def momentum_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top turning-point balls — biggest single-ball win prob swings.
    """
    top = (
        df.dropna(subset=['win_prob_delta'])
        .nlargest(30, 'swing')
        [[
            'match_id', 'batting_team', 'over', 'legal_ball_in_inn',
            'batter', 'bowler', 'runs_total', 'is_wicket',
            'win_prob_prev', 'win_prob', 'win_prob_delta', 'swing',
            'match_phase'
        ]]
        .reset_index(drop=True)
    )
    return top


def venue_analysis(df: pd.DataFrame, match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Chaser win rate and avg win prob per venue.
    Requires match_df to have 'venue' column.
    """
    if 'venue' not in df.columns:
        # Merge from match_df
        venue_map = match_df[['match_id', 'venue']].drop_duplicates()
        df = df.merge(venue_map, on='match_id', how='left')

    match_venue = (
        df.groupby(['match_id', 'venue'])
        .agg(
            result       = ('result', 'first'),
            avg_win_prob = ('win_prob', 'mean'),
            avg_pressure = ('pressure', 'mean'),
        )
        .reset_index()
    )

    venue_stats = (
        match_venue.groupby('venue')
        .agg(
            matches      = ('result', 'count'),
            wins         = ('result', 'sum'),
            avg_win_prob = ('avg_win_prob', 'mean'),
            avg_pressure = ('avg_pressure', 'mean'),
        )
        .reset_index()
    )
    venue_stats['chaser_win_rate'] = venue_stats['wins'] / venue_stats['matches']
    venue_stats = venue_stats[venue_stats['matches'] >= 8]
    venue_stats = venue_stats.sort_values('chaser_win_rate', ascending=False)
    return venue_stats


def toss_analysis(match_df: pd.DataFrame) -> dict:
    """
    Toss decision patterns and their impact on match outcome.
    """
    df = match_df.copy()
    df['toss_won_match'] = (df['toss_winner'] == df['winner']).astype(int)

    overall = df['toss_won_match'].mean()

    by_decision = (
        df.groupby('toss_decision')
        .agg(
            matches       = ('toss_won_match', 'count'),
            toss_win_rate = ('toss_won_match', 'mean'),
        )
        .reset_index()
    )

    by_season = (
        df.groupby('season')
        .agg(
            field_pct = ('toss_decision', lambda x: (x == 'field').mean()),
            bat_pct   = ('toss_decision', lambda x: (x == 'bat').mean()),
        )
        .reset_index()
    )

    return {
        'overall_toss_win_rate' : overall,
        'by_decision'           : by_decision,
        'by_season'             : by_season,
    }


def scoring_analysis(ball_df: pd.DataFrame) -> dict:
    """
    Overall scoring patterns — avg scores, boundary rates, dot ball rates.
    """
    inn1 = ball_df[ball_df['inning'] == 1].copy()

    # Match totals
    totals = (
        inn1.groupby('match_id')['runs_total'].sum()
        .reset_index()
        .rename(columns={'runs_total': 'total'})
    )
    totals = totals.merge(
        ball_df[['match_id', 'team1', 'team2']].drop_duplicates(),
        on='match_id'
    )

    # Boundary rates (all innings)
    legal = ball_df[ball_df['extra_type'].isin(['none', 'legbyes', 'byes', 'penalty'])]
    boundary_rate = (legal['runs_batter'] >= 4).mean()
    six_rate      = (legal['runs_batter'] == 6).mean()
    dot_rate      = (legal['runs_batter'] == 0).mean()

    # Over-wise scoring (innings 1)
    by_over = (
        inn1.groupby('over')['runs_total'].mean()
        .reset_index()
        .rename(columns={'runs_total': 'avg_runs_per_ball'})
    )
    by_over['avg_runs_per_over'] = by_over['avg_runs_per_ball'] * 6

    return {
        'totals'        : totals,
        'boundary_rate' : boundary_rate,
        'six_rate'      : six_rate,
        'dot_rate'      : dot_rate,
        'by_over'       : by_over,
    }


def season_trends(match_df: pd.DataFrame, ball_df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-by-season trends: avg score, chaser win rate, boundary rate.
    """
    inn1 = ball_df[ball_df['inning'] == 1]
    match_scores = (
        inn1.groupby('match_id')['runs_total'].sum()
        .reset_index()
        .rename(columns={'runs_total': 'score'})
    )
    match_scores = match_scores.merge(match_df[['match_id', 'season']], on='match_id')

    # Chaser win (winner == team batting 2nd)
    inn2_teams = (
        ball_df[ball_df['inning'] == 2]
        .groupby('match_id')['batting_team']
        .first()
        .reset_index()
        .rename(columns={'batting_team': 'team2_batting'})
    )
    season_df = match_df.merge(inn2_teams, on='match_id')
    season_df['chaser_won'] = (season_df['winner'] == season_df['team2_batting']).astype(int)

    trend = (
        season_df.groupby('season')
        .agg(matches=('match_id', 'count'), chaser_wins=('chaser_won', 'sum'))
        .reset_index()
    )
    trend['chaser_win_rate'] = trend['chaser_wins'] / trend['matches']

    score_trend = match_scores.groupby('season')['score'].mean().reset_index()
    trend = trend.merge(score_trend, on='season')
    trend = trend.rename(columns={'score': 'avg_first_innings_score'})
    return trend


# ============================================================
#  RECORDS & ADVANCED INSIGHTS
# ============================================================

def batting_records(ball_df: pd.DataFrame, match_df: pd.DataFrame) -> dict:
    """
    Batter leaderboards: runs, boundaries, singles, ducks, fifties/hundreds,
    and players who represented multiple teams.
    """
    legal = ball_df[ball_df['is_legal'] == 1].copy()

    runs = (
        legal.groupby('batter')['runs_batter'].sum()
        .reset_index()
        .rename(columns={'runs_batter': 'total_runs'})
        .sort_values('total_runs', ascending=False)
    )

    sixes = (
        legal[legal['runs_batter'] == 6]
        .groupby('batter').size().reset_index(name='sixes')
        .sort_values('sixes', ascending=False)
    )
    fours = (
        legal[legal['runs_batter'] == 4]
        .groupby('batter').size().reset_index(name='fours')
        .sort_values('fours', ascending=False)
    )
    boundaries = (
        legal[legal['runs_batter'] >= 4]
        .groupby('batter').size().reset_index(name='boundaries')
        .sort_values('boundaries', ascending=False)
    )
    singles = (
        legal[legal['runs_batter'] == 1]
        .groupby('batter').size().reset_index(name='singles')
        .sort_values('singles', ascending=False)
    )
    dots_faced = (
        legal[legal['runs_batter'] == 0]
        .groupby('batter').size().reset_index(name='dot_balls_faced')
        .sort_values('dot_balls_faced', ascending=False)
    )

    batter_innings = (
        legal.groupby(['match_id', 'inning', 'batter'])
        .agg(runs=('runs_batter', 'sum'), dismissed=('is_wicket', 'max'))
        .reset_index()
    )
    ducks = (
        batter_innings[(batter_innings['runs'] == 0) & (batter_innings['dismissed'] == 1)]
        .groupby('batter').size().reset_index(name='ducks')
        .sort_values('ducks', ascending=False)
    )
    fifties = (
        batter_innings[(batter_innings['runs'] >= 50) & (batter_innings['runs'] < 100)]
        .groupby('batter').size().reset_index(name='fifties')
        .sort_values('fifties', ascending=False)
    )
    hundreds = (
        batter_innings[batter_innings['runs'] >= 100]
        .groupby('batter').size().reset_index(name='hundreds')
        .sort_values('hundreds', ascending=False)
    )

    player_teams = (
        ball_df.groupby('batter')['batting_team'].nunique().reset_index()
        .rename(columns={'batting_team': 'teams_count'})
    )
    team_names = (
        ball_df.groupby('batter')['batting_team']
        .apply(lambda x: ', '.join(sorted(x.dropna().unique())))
        .reset_index()
        .rename(columns={'batting_team': 'team_list'})
    )
    multi_team = (
        player_teams.merge(team_names, on='batter')
        .query('teams_count > 1')
        .sort_values('teams_count', ascending=False)
    )

    return {
        'runs': runs,
        'sixes': sixes,
        'fours': fours,
        'boundaries': boundaries,
        'singles': singles,
        'dots_faced': dots_faced,
        'ducks': ducks,
        'fifties': fifties,
        'hundreds': hundreds,
        'multi_team': multi_team,
    }


def bowling_records(ball_df: pd.DataFrame, match_df: pd.DataFrame) -> dict:
    """
    Bowling leaderboards: wickets, economy, dot balls, best figures,
    top wicket-taker by season, and highest over totals conceded.
    """
    legal = ball_df[ball_df['is_legal'] == 1].copy()
    non_bowler_wickets = ['run out', 'retired hurt', 'obstructing the field']

    wickets = (
        legal[(legal['is_wicket'] == 1) & (~legal['dismissal_kind'].isin(non_bowler_wickets))]
        .groupby('bowler').size().reset_index(name='wickets')
        .sort_values('wickets', ascending=False)
    )

    economy = (
        legal.groupby('bowler')
        .agg(runs_conceded=('runs_total', 'sum'), balls_bowled=('is_legal', 'count'))
        .reset_index()
    )
    economy['overs'] = economy['balls_bowled'] / 6
    economy['economy'] = economy['runs_conceded'] / economy['overs']
    economy = economy[economy['overs'] >= 10].sort_values('economy')

    dots_bowled = (
        legal[legal['runs_total'] == 0]
        .groupby('bowler').size().reset_index(name='dot_balls_bowled')
        .sort_values('dot_balls_bowled', ascending=False)
    )

    innings_wkts = (
        legal[(legal['is_wicket'] == 1) & (~legal['dismissal_kind'].isin(non_bowler_wickets))]
        .groupby(['match_id', 'inning', 'bowler'])
        .agg(wickets=('is_wicket', 'sum'))
        .reset_index()
    )
    runs_given = (
        legal.groupby(['match_id', 'inning', 'bowler'])['runs_total']
        .sum().reset_index()
        .rename(columns={'runs_total': 'runs_given'})
    )
    best_figures = (
        innings_wkts.merge(runs_given, on=['match_id', 'inning', 'bowler'])
        .sort_values(['wickets', 'runs_given'], ascending=[False, True])
        .head(20)
    )

    if 'season' in legal.columns:
        season_enriched = legal.copy()
    else:
        season_enriched = legal.merge(match_df[['match_id', 'season']], on='match_id', how='left')
    season_wkts = (
        season_enriched[(season_enriched['is_wicket'] == 1) & (~season_enriched['dismissal_kind'].isin(non_bowler_wickets))]
        .groupby(['season', 'bowler']).size().reset_index(name='wickets')
    )
    season_top = (
        season_wkts.sort_values('wickets', ascending=False)
        .groupby('season').first().reset_index()
    )
    season_top['season_sort'] = pd.to_numeric(season_top['season'], errors='coerce')
    season_top = season_top.sort_values(['season_sort', 'season'], kind='mergesort').drop(columns=['season_sort'])

    over_runs = (
        ball_df.groupby(['match_id', 'inning', 'over'])['runs_total']
        .sum().reset_index()
        .rename(columns={'runs_total': 'runs_in_over'})
    )
    over_teams = ball_df.groupby(['match_id', 'inning', 'over'])['batting_team'].first().reset_index()
    over_runs = (
        over_runs.merge(over_teams, on=['match_id', 'inning', 'over'])
        .sort_values('runs_in_over', ascending=False)
    )

    return {
        'wickets': wickets,
        'economy': economy,
        'dots_bowled': dots_bowled,
        'best_figures': best_figures,
        'season_top': season_top,
        'over_runs': over_runs,
    }


def fielding_records(ball_df: pd.DataFrame) -> dict:
    """
    Fielding leaderboards from available dismissal metadata.
    """
    catches = (
        ball_df[ball_df['dismissal_kind'] == 'caught']
        .groupby('fielder').size().reset_index(name='catches')
        .dropna(subset=['fielder'])
        .sort_values('catches', ascending=False)
    )
    runouts = (
        ball_df[ball_df['dismissal_kind'] == 'run out']
        .groupby('fielder').size().reset_index(name='run_outs')
        .dropna(subset=['fielder'])
        .sort_values('run_outs', ascending=False)
    )
    catches_team = (
        ball_df[ball_df['dismissal_kind'] == 'caught']
        .groupby('bowling_team').size().reset_index(name='catches')
        .dropna(subset=['bowling_team'])
        .sort_values('catches', ascending=False)
    )

    return {
        'catches': catches,
        'runouts': runouts,
        'catches_team': catches_team,
    }


def team_score_records(ball_df: pd.DataFrame, match_df: pd.DataFrame) -> dict:
    """
    Team scoring records and season top batter.
    """
    scores = (
        ball_df.groupby(['match_id', 'inning', 'batting_team'])['runs_total']
        .sum().reset_index()
        .rename(columns={'runs_total': 'score'})
    )
    scores = scores.merge(match_df[['match_id', 'season', 'venue']], on='match_id', how='left')

    highest = scores.sort_values('score', ascending=False).head(20)
    lowest = scores[scores['score'] > 10].sort_values('score').head(20)

    legal = ball_df[ball_df['is_legal'] == 1].copy()
    if 'season' not in legal.columns:
        legal = legal.merge(match_df[['match_id', 'season']], on='match_id', how='left')

    season_runs = (
        legal
        .groupby(['season', 'batter'])['runs_batter']
        .sum().reset_index()
    )
    season_top_batter = (
        season_runs.sort_values('runs_batter', ascending=False)
        .groupby('season').first().reset_index()
        .rename(columns={'runs_batter': 'runs'})
    )
    season_top_batter['season_sort'] = pd.to_numeric(season_top_batter['season'], errors='coerce')
    season_top_batter = season_top_batter.sort_values(['season_sort', 'season'], kind='mergesort').drop(columns=['season_sort'])

    return {
        'highest': highest,
        'lowest': lowest,
        'season_top_batter': season_top_batter,
    }


def match_records(match_df: pd.DataFrame, ball_df: pd.DataFrame) -> dict:
    """
    Team wins, playoff appearances, trophies, and captain proxy stats.
    """
    all_matches = pd.concat([
        match_df[['match_id', 'team1', 'winner', 'season', 'is_playoff']].rename(columns={'team1': 'team'}),
        match_df[['match_id', 'team2', 'winner', 'season', 'is_playoff']].rename(columns={'team2': 'team'}),
    ]).drop_duplicates(subset=['match_id', 'team'])
    all_matches['won'] = (all_matches['team'] == all_matches['winner']).astype(int)

    team_wins = (
        all_matches.groupby('team')
        .agg(matches=('match_id', 'count'), wins=('won', 'sum'))
        .reset_index()
    )
    team_wins['losses'] = team_wins['matches'] - team_wins['wins']
    team_wins['win_rate'] = team_wins['wins'] / team_wins['matches']
    team_wins = team_wins[team_wins['matches'] >= 10].sort_values('wins', ascending=False)

    playoffs = (
        all_matches[all_matches['is_playoff'] == True]
        .groupby('team').size().reset_index(name='playoff_appearances')
        .sort_values('playoff_appearances', ascending=False)
    )

    finals = match_df[match_df['stage'].fillna('').str.lower().str.contains('final', na=False)]
    trophies = (
        finals.groupby('winner').size().reset_index(name='trophies')
        .sort_values('trophies', ascending=False)
    )

    cap1 = match_df[['match_id', 'team1', 'captain_team1', 'winner']].rename(
        columns={'team1': 'team', 'captain_team1': 'captain'}
    )
    cap2 = match_df[['match_id', 'team2', 'captain_team2', 'winner']].rename(
        columns={'team2': 'team', 'captain_team2': 'captain'}
    )
    cap_df = pd.concat([cap1, cap2]).dropna(subset=['captain'])
    cap_df['won'] = (cap_df['team'] == cap_df['winner']).astype(int)
    cap_stats = (
        cap_df.groupby('captain')
        .agg(matches=('match_id', 'count'), wins=('won', 'sum'))
        .reset_index()
    )
    cap_stats = cap_stats[cap_stats['matches'] >= 5].sort_values('wins', ascending=False)
    cap_stats['win_rate'] = cap_stats['wins'] / cap_stats['matches']

    return {
        'team_wins': team_wins,
        'playoffs': playoffs,
        'trophies': trophies,
        'cap_stats': cap_stats,
    }
