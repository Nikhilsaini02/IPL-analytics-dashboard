# ============================================================
#  charts.py
#  All chart functions — each returns a matplotlib Figure.
#  Used by both standalone scripts and Streamlit app.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ── THEME ────────────────────────────────────────────────────
BG       = '#03112a'
PANEL    = '#0a2348'
BORDER   = '#1e4378'
TEXT     = '#deebff'
SUBTEXT  = '#a9c2ea'
WHITE    = '#f8fbff'
PURPLE   = '#f6c453'
TEAL     = '#42c9ff'
CORAL    = '#ff8a4a'
AMBER    = '#ffd166'
RED      = '#ff5b6e'
GREEN    = '#7ee18d'
MUTED    = '#5f80b4'
TEAM_COLORS = [PURPLE, TEAL, CORAL, AMBER, GREEN, RED, '#8cd4ff', '#6fe2c8',
               '#f9c46b', '#ffa96b', '#9ed3f4', '#c9e67c']


def _style(ax, title='', xlabel='', ylabel='', grid_axis='y'):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    if title:
        ax.set_title(title, color=WHITE, pad=9, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.grid(axis=grid_axis, color=BORDER, linestyle='--', alpha=0.6)


def _fig(w=14, h=6, n_rows=1, n_cols=1, **kwargs):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h),
                             facecolor=BG, **kwargs)
    return fig, axes


# ============================================================
#  1. MATCH REPLAYER
# ============================================================

def chart_match_replayer(match_df: pd.DataFrame) -> plt.Figure:
    """
    Ball-by-ball win probability + runs + momentum for one match.
    win_prob column = historical win% (pure lookup, no model).
    """
    bat_team  = match_df['batting_team'].iloc[0]
    team1     = match_df['team1'].iloc[0]
    bowl_team = team1 if team1 != bat_team else match_df['team2'].iloc[0]
    target    = match_df['target'].iloc[0]
    winner    = match_df['winner'].iloc[0]
    final_runs = match_df['runs_scored'].iloc[-1]
    final_wkts = match_df['wickets_fallen'].iloc[-1]

    result_str = (
        f"{bat_team} won by {10 - int(final_wkts)} wkt(s)"
        if winner == bat_team
        else f"{bowl_team} won by {int(target) - int(final_runs) - 1} run(s)"
    )

    balls  = match_df['legal_ball_in_inn'].values
    probs  = match_df['win_prob'].ffill().fillna(0.5).values
    runs   = match_df['runs_total'].values
    deltas = match_df['win_prob_delta'].fillna(0).values

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             facecolor=BG,
                             gridspec_kw={'height_ratios': [5, 1.5, 1.5]})
    fig.suptitle(
        f'{bat_team}  vs  {bowl_team}   |   Target: {int(target)}   |   {result_str}',
        fontsize=13, fontweight='bold', color=WHITE, y=1.01
    )

    # ── Panel 1: Win prob ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)

    # Phase shading
    ax.axvspan(1,  36,  alpha=0.05, color=TEAL)
    ax.axvspan(37, 90,  alpha=0.05, color=PURPLE)
    ax.axvspan(91, 120, alpha=0.05, color=CORAL)
    for xp, lbl, col in [(18, 'POWERPLAY', TEAL), (63, 'MIDDLE', PURPLE), (105, 'DEATH', CORAL)]:
        if xp <= max(balls):
            ax.text(xp, 0.97, lbl, ha='center', va='top',
                    fontsize=7, color=col, alpha=0.65, fontweight='bold')

    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle='--', alpha=0.8)
    ax.text(1.5, 0.515, '50%', fontsize=8, color=MUTED)

    ax.fill_between(balls, probs, 0.5,
                    where=(probs >= 0.5), alpha=0.22, color=TEAL, interpolate=True)
    ax.fill_between(balls, probs, 0.5,
                    where=(probs < 0.5),  alpha=0.22, color=CORAL, interpolate=True)
    ax.plot(balls, probs, color=WHITE, linewidth=2, alpha=0.9)

    # Over lines
    for ov in range(6, int(max(balls)), 6):
        ax.axvline(ov, color=MUTED, linewidth=0.35, alpha=0.4)

    # Wickets
    for _, row in match_df[match_df['is_wicket'] == 1].iterrows():
        b, p = row['legal_ball_in_inn'], row['win_prob']
        if pd.isna(p):
            continue
        ax.scatter(b, p, color=RED, s=85, zorder=6, marker='v')
        ax.annotate(
            f"W{int(row['wickets_fallen'])}",
            xy=(b, p),
            xytext=(b + 1.5, min(p + 0.06, 0.95)),
            fontsize=7.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=0.8),
            zorder=7
        )

    # Sixes
    for _, row in match_df[match_df['runs_batter'] == 6].iterrows():
        b, p = row['legal_ball_in_inn'], row['win_prob']
        if not pd.isna(p):
            ax.scatter(b, p, color=AMBER, s=50, zorder=5, marker='*')

    # Turning point
    if len(deltas) > 1:
        idx = np.nanargmax(np.abs(deltas[1:])) + 1
        b_, p_, d_ = balls[idx], probs[idx], deltas[idx]
        ax.annotate(
            f'Turning point\n{"↑" if d_ > 0 else "↓"}{abs(d_):.2f}',
            xy=(b_, p_),
            xytext=(b_ + 4, p_ + (0.12 if p_ < 0.75 else -0.15)),
            fontsize=8, color=AMBER, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL,
                      edgecolor=AMBER, alpha=0.85),
            zorder=8
        )

    ax.set_xlim(1, max(balls) + 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Win Probability (chasing team)', fontsize=9, color=TEXT)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks(range(6, int(max(balls)) + 6, 6))
    ax.set_xticklabels([f'Ov {i//6}' for i in range(6, int(max(balls)) + 6, 6)],
                       fontsize=8, color=SUBTEXT)
    ax.tick_params(colors=SUBTEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)

    legend_els = [
        Line2D([0], [0], color=WHITE, linewidth=2, label='Win probability'),
        Line2D([0], [0], color=RED,   marker='v', linewidth=0, markersize=8, label='Wicket'),
        Line2D([0], [0], color=AMBER, marker='*', linewidth=0, markersize=9, label='Six'),
        mpatches.Patch(color=TEAL,  alpha=0.4, label=f'{bat_team} ahead'),
        mpatches.Patch(color=CORAL, alpha=0.4, label=f'{bowl_team} ahead'),
    ]
    ax.legend(handles=legend_els, loc='upper left', framealpha=0.2,
              fontsize=8, ncol=3, facecolor=PANEL)

    # ── Panel 2: Runs per ball ────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    wkts = match_df['is_wicket'].values
    bar_cols = [RED if w else (AMBER if r == 6 else (GREEN if r == 4 else TEAL))
                for r, w in zip(runs, wkts)]
    ax2.bar(balls, runs, color=bar_cols, alpha=0.85, width=0.7)
    ax2.set_xlim(1, max(balls) + 1)
    ax2.set_ylabel('Runs/Ball', fontsize=9, color=TEXT)
    ax2.set_ylim(0, 8)
    ax2.set_xticks([])
    ax2.tick_params(colors=SUBTEXT)
    ax2.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor(BORDER)
    run_leg = [mpatches.Patch(color=AMBER, label='6'),
               mpatches.Patch(color=GREEN, label='4'),
               mpatches.Patch(color=TEAL,  label='1–3'),
               mpatches.Patch(color=RED,   label='Wicket ball')]
    ax2.legend(handles=run_leg, loc='upper right', framealpha=0.2,
               fontsize=7.5, ncol=4, facecolor=PANEL)

    # ── Panel 3: Momentum (Δwin prob) ────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor(PANEL)
    dcols = [TEAL if d >= 0 else CORAL for d in deltas]
    ax3.bar(balls, deltas, color=dcols, alpha=0.8, width=0.7)
    ax3.axhline(0, color=MUTED, linewidth=0.8)
    ax3.set_xlim(1, max(balls) + 1)
    ax3.set_ylabel('Δ Win Prob', fontsize=9, color=TEXT)
    ax3.set_xlabel('Legal ball number', fontsize=9, color=TEXT)
    ax3.set_xticks(range(6, int(max(balls)) + 6, 6))
    ax3.set_xticklabels([f'Ov {i//6}' for i in range(6, int(max(balls)) + 6, 6)],
                        fontsize=8, color=SUBTEXT)
    ax3.tick_params(colors=SUBTEXT)
    ax3.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)
    for sp in ax3.spines.values():
        sp.set_edgecolor(BORDER)

    plt.tight_layout()
    return fig


# ============================================================
#  2. PHASE ANALYSIS CHARTS
# ============================================================

def chart_phase_overview(phase_data: dict) -> plt.Figure:
    by_over  = phase_data['by_over']
    by_phase = phase_data['by_phase']

    fig, axes = _fig(16, 6, 1, 2)
    axes = axes if hasattr(axes, '__len__') else [axes]

    # Over-wise win prob
    ax = axes[0]
    ax.set_facecolor(PANEL)
    cols = [TEAL if o <= 6 else (PURPLE if o <= 15 else CORAL)
            for o in by_over['over']]
    ax.bar(by_over['over'], by_over['avg_win_prob'],
           color=cols, alpha=0.85, width=0.75)
    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax.axvline(6.5,  color=TEAL,   linestyle=':', alpha=0.5)
    ax.axvline(15.5, color=PURPLE, linestyle=':', alpha=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(ax, 'Avg Win Probability by Over', 'Over', 'Avg Win Probability')
    leg = [mpatches.Patch(color=TEAL,   label='Powerplay'),
           mpatches.Patch(color=PURPLE, label='Middle'),
           mpatches.Patch(color=CORAL,  label='Death')]
    ax.legend(handles=leg, framealpha=0.2, fontsize=9, facecolor=PANEL)

    # Volatility by over
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    ax2.plot(by_over['over'], by_over['avg_swing'],
             color=AMBER, linewidth=2.2, marker='o', markersize=4)
    ax2.fill_between(by_over['over'], by_over['avg_swing'],
                     alpha=0.18, color=AMBER)
    ax2.axvline(6.5,  color=TEAL,   linestyle=':', alpha=0.5)
    ax2.axvline(15.5, color=PURPLE, linestyle=':', alpha=0.5)
    _style(ax2, 'Win Prob Volatility by Over\n(avg |Δ prob| per ball)',
           'Over', 'Avg |Δ Win Prob|', grid_axis='both')

    fig.tight_layout()
    return fig


# ============================================================
#  3. TEAM ANALYSIS CHARTS
# ============================================================

def chart_team_win_rates(team_stats: pd.DataFrame) -> plt.Figure:
    fig, axes = _fig(16, 6, 1, 2)

    ts = team_stats.sort_values('win_rate')

    # Win rate
    ax = axes[0]
    ax.set_facecolor(PANEL)
    cols = [TEAL if v > 0.5 else CORAL for v in ts['win_rate']]
    bars = ax.barh(ts['batting_team'], ts['win_rate'], color=cols, alpha=0.85, height=0.6)
    ax.axvline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    for bar, val in zip(bars, ts['win_rate']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', fontsize=8, color=WHITE)
    _style(ax, 'Chase Win Rate by Team', 'Win Rate', '', grid_axis='x')

    # Volatility
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    tv = team_stats.sort_values('volatility')
    ax2.barh(tv['batting_team'], tv['volatility'], color=PURPLE, alpha=0.8, height=0.6)
    _style(ax2, 'Chase Volatility by Team\n(avg |Δ win prob| per ball — lower = calmer)',
           'Volatility', '', grid_axis='x')

    fig.tight_layout()
    return fig


def chart_team_journeys(journey_df: pd.DataFrame, teams: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(PANEL)

    for i, team in enumerate(teams):
        sub = journey_df[journey_df['batting_team'] == team]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        ax.plot(sub['over'], sub['avg_win_prob'],
                linewidth=2.5, color=color, label=team,
                marker='o', markersize=3.5, alpha=0.9)

    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax.axvline(6.5,  color=MUTED, linestyle=':', alpha=0.4)
    ax.axvline(15.5, color=MUTED, linestyle=':', alpha=0.4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    for xp, lbl in [(3.5, 'Powerplay'), (11, 'Middle'), (18, 'Death')]:
        ax.text(xp, 0.97, lbl, ha='center', fontsize=8, color=SUBTEXT)

    _style(ax, 'Average Win Probability Journey by Team (over-by-over)',
           'Over', 'Avg Win Probability', grid_axis='both')
    ax.set_xlim(1, 20)
    ax.set_ylim(0.1, 1.0)
    ax.legend(framealpha=0.2, fontsize=9, facecolor=PANEL, ncol=2)
    fig.tight_layout()
    return fig


# ============================================================
#  4. WICKET IMPACT CHART
# ============================================================

def chart_wicket_impact(wicket_df: pd.DataFrame) -> plt.Figure:
    fig, axes = _fig(16, 6, 1, 2)

    wi = wicket_df.sort_values('wicket_number')
    peak = wi.loc[wi['prob_drop'].idxmax()]

    # Drop per wicket number
    ax = axes[0]
    ax.set_facecolor(PANEL)
    wcols = [RED if w == peak['wicket_number'] else CORAL for w in wi['wicket_number']]
    ax.bar(wi['wicket_number'].astype(int), wi['prob_drop'],
           color=wcols, alpha=0.85, width=0.6)
    ax.set_xticks(wi['wicket_number'].astype(int))
    ax.set_xticklabels([f'#{i}' for i in wi['wicket_number'].astype(int)],
                       color=SUBTEXT)
    ax.annotate(
        f'Most clutch\n#{int(peak["wicket_number"])}',
        xy=(peak['wicket_number'], peak['prob_drop']),
        xytext=(peak['wicket_number'] + 1.2, peak['prob_drop'] * 0.82),
        fontsize=8.5, color=RED, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2)
    )
    _style(ax, 'Avg Win Prob Drop per Wicket Fall\n(chasing team perspective)',
           'Wicket Number', 'Avg Win Prob Drop')

    # Before vs after
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    x = wi['wicket_number'].astype(int)
    ax2.plot(x, wi['avg_prob_before'], 'o-', color=TEAL,  linewidth=2,
             markersize=5, label='Before wicket')
    ax2.plot(x, wi['avg_prob_after'],  's-', color=CORAL, linewidth=2,
             markersize=5, label='After wicket')
    ax2.fill_between(x, wi['avg_prob_before'], wi['avg_prob_after'],
                     alpha=0.12, color=RED)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(framealpha=0.2, fontsize=9, facecolor=PANEL)
    _style(ax2, 'Win Prob Before vs After Each Wicket',
           'Wicket Number', 'Win Probability', grid_axis='both')

    fig.tight_layout()
    return fig


# ============================================================
#  5. VENUE CHART
# ============================================================

def chart_venue(venue_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    top = pd.concat([venue_df.head(top_n // 2), venue_df.tail(top_n // 2)])
    top = top.drop_duplicates().sort_values('chaser_win_rate')

    cols = [TEAL if v > 0.5 else CORAL for v in top['chaser_win_rate']]
    bars = ax.barh(top['venue'], top['chaser_win_rate'], color=cols, alpha=0.85, height=0.6)
    ax.axvline(0.5, color=MUTED, linestyle='--', linewidth=1)
    for bar, val in zip(bars, top['chaser_win_rate']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', fontsize=8, color=WHITE)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(ax, 'Chaser Win Rate by Venue\n(green = chaser-friendly | red = bowling-friendly)',
           'Chaser Win Rate', '', grid_axis='x')
    leg = [mpatches.Patch(color=TEAL,  label='Chaser-friendly (>50%)'),
           mpatches.Patch(color=CORAL, label='Bowling-friendly (<50%)')]
    ax.legend(handles=leg, framealpha=0.2, fontsize=9, facecolor=PANEL)

    fig.tight_layout()
    return fig


# ============================================================
#  6. TOSS ANALYSIS CHARTS
# ============================================================

def chart_toss(toss_data: dict) -> plt.Figure:
    fig, axes = _fig(16, 6, 1, 2)

    # Toss decision by season
    ax = axes[0]
    ax.set_facecolor(PANEL)
    bs = toss_data['by_season']
    ax.bar(bs['season'].astype(str), bs['field_pct'],
           label='Field first', color=TEAL, alpha=0.85, width=0.6)
    ax.bar(bs['season'].astype(str), bs['bat_pct'],
           bottom=bs['field_pct'], label='Bat first', color=CORAL, alpha=0.85, width=0.6)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xticklabels(bs['season'].astype(str), rotation=45, ha='right', color=SUBTEXT)
    ax.legend(framealpha=0.2, fontsize=9, facecolor=PANEL)
    _style(ax, 'Toss Decision Trends by Season', 'Season', 'Proportion')

    # Toss win rate by decision
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    bd = toss_data['by_decision']
    cols = [TEAL, CORAL]
    bars = ax2.bar(bd['toss_decision'], bd['toss_win_rate'],
                   color=cols, alpha=0.85, width=0.4)
    ax2.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    for bar, val in zip(bars, bd['toss_win_rate']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.1%}', ha='center', fontsize=10, color=WHITE, fontweight='bold')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim(0, 0.75)
    _style(ax2, f'Match Win Rate by Toss Decision\n(overall toss advantage: {toss_data["overall_toss_win_rate"]:.1%})',
           'Decision', 'Win Rate when Toss is Won', grid_axis='y')

    fig.tight_layout()
    return fig


# ============================================================
#  7. SCORING PATTERNS
# ============================================================

def chart_scoring(scoring_data: dict) -> plt.Figure:
    fig, axes = _fig(16, 6, 1, 2)

    # Over-wise avg runs
    ax = axes[0]
    ax.set_facecolor(PANEL)
    by_over = scoring_data['by_over']
    cols = [TEAL if o <= 6 else (PURPLE if o <= 15 else CORAL) for o in by_over['over']]
    ax.bar(by_over['over'], by_over['avg_runs_per_over'],
           color=cols, alpha=0.85, width=0.75)
    _style(ax, 'Avg Runs per Over (1st innings)', 'Over', 'Avg Runs per Over')

    # Summary stats
    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    labels = ['Boundary\nball %', 'Six\nball %', 'Dot\nball %']
    values = [
        scoring_data['boundary_rate'],
        scoring_data['six_rate'],
        scoring_data['dot_rate'],
    ]
    bar_cols = [TEAL, AMBER, CORAL]
    bars = ax2.bar(labels, values, color=bar_cols, alpha=0.85, width=0.45)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.1%}', ha='center', fontsize=11, color=WHITE, fontweight='bold')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(ax2, 'Ball Outcome Rates (all legal deliveries)', '', '%')

    fig.tight_layout()
    return fig


# ============================================================
#  8. SEASON TRENDS
# ============================================================

def chart_season_trends(trend_df: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax2 = ax1.twinx()

    ax1.set_facecolor(PANEL)
    ax2.set_facecolor(PANEL)

    x = trend_df['season'].astype(str)

    ax1.bar(x, trend_df['avg_first_innings_score'],
            color=PURPLE, alpha=0.5, width=0.5, label='Avg 1st innings score')
    ax2.plot(x, trend_df['chaser_win_rate'],
             color=TEAL, linewidth=2.5, marker='o', markersize=6, label='Chaser win rate')
    ax2.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    ax1.set_xlabel('Season', color=TEXT, fontsize=9)
    ax1.set_ylabel('Avg 1st Innings Score', color=PURPLE, fontsize=9)
    ax2.set_ylabel('Chaser Win Rate', color=TEAL, fontsize=9)
    ax1.set_title('IPL Season Trends — Avg Score vs Chaser Win Rate',
                  color=WHITE, pad=9, fontsize=11)
    ax1.tick_params(axis='x', colors=SUBTEXT, rotation=45)
    ax1.tick_params(axis='y', colors=SUBTEXT)
    ax2.tick_params(axis='y', colors=SUBTEXT)
    for sp in ax1.spines.values():
        sp.set_edgecolor(BORDER)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               framealpha=0.2, fontsize=9, facecolor=PANEL)
    ax1.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)

    fig.tight_layout()
    return fig


# ============================================================
#  9. WIN PROBABILITY HEATMAP
# ============================================================

def chart_win_prob_heatmap(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(PANEL)

    pivot = (
        df[df['balls_left'] > 0]
        .groupby([
            'wickets_left',
            pd.cut(df[df['balls_left'] > 0]['balls_left'],
                   bins=[0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                   labels=['≤12', '13–24', '25–36', '37–48', '49–60',
                            '61–72', '73–84', '85–96', '97–108', '109–120'])
        ])['win_prob']
        .mean()
        .unstack()
    )

    if pivot.empty:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center',
                color=TEXT, transform=ax.transAxes)
        return fig

    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, origin='lower')

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=SUBTEXT, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color=SUBTEXT, fontsize=8.5, rotation=30, ha='right')

    ax.set_title('Historical Win Probability — Wickets Left × Balls Left\n(green = chasing team likely to win | red = likely to lose)',
                 color=WHITE, pad=9, fontsize=11)
    ax.set_xlabel('Balls Left (binned)', color=TEXT, fontsize=9)
    ax.set_ylabel('Wickets Left', color=TEXT, fontsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=SUBTEXT)
    cbar.set_label('Historical Win Probability', color=TEXT, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=7.5,
                        color='#1a1d27' if 0.3 < val < 0.7 else WHITE)

    fig.tight_layout()
    return fig


# ============================================================
#  10. PHASE DISTRIBUTION (BOX + SCATTER)
# ============================================================

def chart_phase_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Distribution-focused phase view: boxplot + jittered swing scatter.
    """
    fig, axes = _fig(16, 6, 1, 2)

    order = ['Powerplay', 'Middle', 'Death']
    pdata = [df[df['match_phase'] == p]['win_prob'].dropna().values for p in order]

    ax = axes[0]
    ax.set_facecolor(PANEL)
    bp = ax.boxplot(
        pdata,
        labels=order,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=WHITE, linewidth=2),
    )
    for box, col in zip(bp['boxes'], [TEAL, PURPLE, CORAL]):
        box.set_facecolor(col)
        box.set_alpha(0.35)
        box.set_edgecolor(col)
        box.set_linewidth(1.4)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(ax, 'Win Probability Distribution by Phase', 'Phase', 'Win Probability')

    ax2 = axes[1]
    ax2.set_facecolor(PANEL)
    sample = df[['match_phase', 'swing']].dropna()
    if len(sample) > 18000:
        sample = sample.sample(18000, random_state=42)

    phase_to_x = {'Powerplay': 1, 'Middle': 2, 'Death': 3}
    xs = sample['match_phase'].map(phase_to_x).astype(float).values
    jitter = np.random.default_rng(42).normal(0, 0.06, len(sample))
    xj = xs + jitter

    cols = sample['match_phase'].map({'Powerplay': TEAL, 'Middle': PURPLE, 'Death': CORAL}).values
    ax2.scatter(xj, sample['swing'].values, s=10, c=cols, alpha=0.20, edgecolors='none')

    phase_means = sample.groupby('match_phase')['swing'].mean().reindex(order)
    ax2.plot([1, 2, 3], phase_means.values, color=AMBER, marker='o', linewidth=2)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(order, color=SUBTEXT)
    _style(ax2, 'Ball-Level Volatility by Phase (Jitter Plot)', 'Phase', 'Swing |Δ Win Prob|')

    fig.tight_layout()
    return fig


# ============================================================
#  11. TEAM BUBBLE MAP
# ============================================================

def chart_team_performance_bubble(team_stats: pd.DataFrame) -> plt.Figure:
    """
    Bubble chart: win rate vs volatility, sized by matches.
    """
    fig, ax = plt.subplots(figsize=(14, 6.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ts = team_stats.copy()
    if ts.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', color=TEXT, transform=ax.transAxes)
        return fig

    sizes = np.clip(ts['matches'].values * 8, 90, 950)
    sc = ax.scatter(
        ts['volatility'],
        ts['win_rate'],
        s=sizes,
        c=ts['avg_pressure'],
        cmap='coolwarm',
        alpha=0.72,
        edgecolors=WHITE,
        linewidths=0.7,
    )

    best = ts.nlargest(min(6, len(ts)), 'win_rate')
    for _, row in best.iterrows():
        ax.text(
            row['volatility'] + 0.0004,
            row['win_rate'] + 0.002,
            row['batting_team'],
            fontsize=8,
            color=WHITE,
        )

    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(
        ax,
        'Team Performance Map (bubble size = matches)',
        'Volatility (avg |Δ win prob| per ball)',
        'Chase Win Rate',
        grid_axis='both',
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors=SUBTEXT)
    cbar.set_label('Avg Pressure (RRR - CRR)', color=TEXT, fontsize=9)

    fig.tight_layout()
    return fig


# ============================================================
#  12. VENUE BUBBLE MAP
# ============================================================

def chart_venue_bubble(venue_df: pd.DataFrame) -> plt.Figure:
    """
    Bubble map: venue sample size vs chase win rate.
    """
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(PANEL)

    vd = venue_df.copy()
    if vd.empty:
        ax.text(0.5, 0.5, 'No venue data available', ha='center', va='center', color=TEXT, transform=ax.transAxes)
        return fig

    sizes = np.clip(vd['matches'].values * 9, 70, 900)
    colors = np.where(vd['chaser_win_rate'] >= 0.5, TEAL, CORAL)

    ax.scatter(
        vd['matches'],
        vd['chaser_win_rate'],
        s=sizes,
        c=colors,
        alpha=0.55,
        edgecolors=WHITE,
        linewidths=0.6,
    )

    top = vd.nlargest(min(8, len(vd)), 'matches')
    for _, row in top.iterrows():
        ax.text(row['matches'] + 0.35, row['chaser_win_rate'] + 0.003, row['venue'][:20], fontsize=7.5, color=WHITE)

    ax.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    _style(
        ax,
        'Venue Landscape (bubble size = number of matches)',
        'Matches at Venue',
        'Chaser Win Rate',
        grid_axis='both',
    )

    fig.tight_layout()
    return fig


# ============================================================
#  13. SEASON FLOW (AREA)
# ============================================================

def chart_season_flow(trend_df: pd.DataFrame) -> plt.Figure:
    """
    Area + line chart for season-level direction.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax1.set_facecolor(PANEL)
    ax2 = ax1.twinx()
    ax2.set_facecolor(PANEL)

    td = trend_df.copy()
    td['season_sort'] = pd.to_numeric(td['season'], errors='coerce')
    td = td.sort_values(['season_sort', 'season'], kind='mergesort').drop(columns=['season_sort'])
    x = np.arange(len(td))
    xlbl = td['season'].astype(str).tolist()

    score = td['avg_first_innings_score'].values
    chase = td['chaser_win_rate'].values

    ax1.fill_between(x, score, color=PURPLE, alpha=0.28)
    ax1.plot(x, score, color=PURPLE, linewidth=2.5, marker='o', markersize=4)

    ax2.plot(x, chase, color=TEAL, linewidth=2.5, marker='o', markersize=4)
    ax2.fill_between(x, chase, color=TEAL, alpha=0.14)
    ax2.axhline(0.5, color=MUTED, linestyle='--', linewidth=1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlbl, rotation=45, ha='right', color=SUBTEXT)
    ax1.set_ylabel('Avg 1st Innings Score', color=PURPLE)
    ax2.set_ylabel('Chaser Win Rate', color=TEAL)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax1.set_title('Season Flow: Scoring Pressure vs Chase Success', color=WHITE, pad=10, fontsize=11)
    ax1.grid(axis='y', color=BORDER, linestyle='--', alpha=0.5)

    for sp in ax1.spines.values():
        sp.set_edgecolor(BORDER)
    ax1.tick_params(axis='y', colors=SUBTEXT)
    ax2.tick_params(axis='y', colors=SUBTEXT)

    fig.tight_layout()
    return fig


# ============================================================
#  14. TURNING POINT SCATTER
# ============================================================

def chart_turning_points_scatter(momentum_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter chart for top turning points with direction encoding.
    """
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(PANEL)

    md = momentum_df.copy().dropna(subset=['win_prob_delta', 'swing'])
    if md.empty:
        ax.text(0.5, 0.5, 'No turning point data', ha='center', va='center', color=TEXT, transform=ax.transAxes)
        return fig

    md = md.sort_values('swing', ascending=False).head(40).reset_index(drop=True)
    md['rank'] = np.arange(1, len(md) + 1)

    colors = np.where(md['win_prob_delta'] >= 0, TEAL, CORAL)
    ax.scatter(md['rank'], md['win_prob_delta'], s=md['swing'] * 5200, c=colors, alpha=0.65,
               edgecolors=WHITE, linewidths=0.6)
    ax.axhline(0, color=MUTED, linewidth=1)

    for _, row in md.head(8).iterrows():
        label = f"{row['batting_team'][:7]} Ov{int(row['over'])}"
        ax.text(row['rank'] + 0.2, row['win_prob_delta'] + 0.001, label, fontsize=7.5, color=WHITE)

    _style(
        ax,
        'Turning Points Map (bubble size = absolute swing)',
        'Top Event Rank (by swing)',
        'Win Probability Delta',
        grid_axis='both',
    )

    fig.tight_layout()
    return fig


# ============================================================
#  15. RECORDS HELPERS
# ============================================================

def chart_record_barh(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    top_n: int = 15,
    color: str = PURPLE,
) -> plt.Figure:
    """
    Reusable horizontal leaderboard bar chart.
    """
    plot_df = data_df.head(top_n).sort_values(x_col)
    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.45)), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.barh(plot_df[y_col].astype(str), plot_df[x_col], color=color, alpha=0.85, height=0.65)
    for i, val in enumerate(plot_df[x_col]):
        label = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
        ax.text(val, i, f"  {label}", va='center', fontsize=8.5, color=WHITE)

    _style(ax, title, xlabel, '', grid_axis='x')
    fig.tight_layout()
    return fig


def chart_record_barv(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    top_n: int = 15,
    color: str = PURPLE,
) -> plt.Figure:
    """
    Reusable vertical leaderboard bar chart.
    """
    plot_df = data_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
    ax.set_facecolor(PANEL)

    ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color=color, alpha=0.85, width=0.65)
    for i, val in enumerate(plot_df[y_col]):
        label = f"{val:,.2f}" if isinstance(val, float) else f"{val:,}"
        ax.text(i, val, label, ha='center', fontsize=8, color=WHITE)

    ax.tick_params(axis='x', rotation=45)
    _style(ax, title, '', ylabel)
    fig.tight_layout()
    return fig
