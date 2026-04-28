import { ChangeDetectionStrategy, Component, Input, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatchPrediction } from '../../services/prediction.service';
import { TeamCrestComponent } from './team-crest.component';

@Component({
  selector: 'app-recent-result',
  standalone: true,
  imports: [CommonModule, TeamCrestComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <article class="card">
      <header class="head">
        <span class="date">{{ match.date | date: 'EEE, MMM d' }}</span>
        <div class="match-line">
          <span class="team home">
            <app-team-crest class="mini-crest" [teamId]="match.home_team_id" [name]="match.home_team_name" />
            <span class="team-name">{{ match.home_team_name }}</span>
          </span>
          <span class="sep" aria-hidden="true">·</span>
          <span class="team away">
            <span class="team-name">{{ match.away_team_name }}</span>
            <app-team-crest class="mini-crest" [teamId]="match.away_team_id" [name]="match.away_team_name" />
          </span>
        </div>
      </header>

      <div class="body">
        <div class="left">
          <div class="row">
            <span class="lbl">ACTUAL</span>
            <span class="value">[{{ actualParts().home }}]-[{{ actualParts().away }}]</span>
          </div>
          <div class="row">
            <span class="lbl predicted-lbl">PREDICTED</span>
            <span class="value">[{{ predictedParts().home }}]-[{{ predictedParts().away }}]</span>
          </div>
        </div>
        <div class="gauge">
          <svg viewBox="0 0 64 64" width="64" height="64" aria-hidden="true">
            <circle cx="32" cy="32" r="28" stroke="rgba(255,255,255,0.15)" stroke-width="6" fill="none" />
            <circle
              cx="32" cy="32" r="28"
              [attr.stroke]="gaugeColor()"
              stroke-width="6"
              fill="none"
              stroke-linecap="round"
              stroke-dasharray="175.93"
              [attr.stroke-dashoffset]="175.93 - 175.93 * accFraction()"
              transform="rotate(-90 32 32)"
            />
          </svg>
          <div class="gauge-text">
            <span class="pct">{{ accPct() }}%</span>
            <span class="acc-lbl">accuracy</span>
          </div>
        </div>
      </div>

      <footer class="foot">{{ rationale() }}</footer>
    </article>
  `,
  styles: `
    .card {
      background: var(--ericfc-navy);
      color: #fff;
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 6px 18px rgba(27, 51, 88, 0.18);
      display: flex;
      flex-direction: column;
      border: 1px solid rgba(194, 155, 81, 0.25);
      transition: transform .18s ease, box-shadow .18s ease;
      will-change: transform;
    }
    @media (hover: hover) and (min-width: 720px) {
      .card:hover {
        transform: scale(1.02);
        box-shadow:
          0 10px 28px rgba(27, 51, 88, 0.28),
          0 0 22px rgba(194, 155, 81, 0.28);
      }
    }
    .head {
      background: var(--ericfc-navy-700);
      padding: 8px 14px 10px;
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: #fff;
    }
    .head .date {
      font-size: 0.66rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--ericfc-gold-300);
    }
    .head .match-line {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      font-weight: 700;
    }
    .head .team {
      flex: 1;
      min-width: 0;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .head .team.away { justify-content: flex-end; }
    .head .team .team-name {
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .head .mini-crest { --crest-size: 22px; }
    .sep { color: var(--ericfc-gold); font-weight: 700; flex-shrink: 0; }
    .body {
      padding: 12px 14px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .left {
      flex: 1;
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .row {
      display: flex;
      gap: 8px;
      align-items: baseline;
      font-size: 0.82rem;
    }
    .lbl {
      font-weight: 700;
      letter-spacing: 0.1em;
      color: var(--ericfc-gold-300);
      min-width: 76px;
    }
    .predicted-lbl { color: rgba(255, 255, 255, 0.85); }
    .value {
      font-family: 'Roboto Condensed', 'Inter', sans-serif;
      font-weight: 800;
      font-size: 0.96rem;
      letter-spacing: 0.02em;
      color: #fff;
    }
    .gauge {
      position: relative;
      width: 64px;
      height: 64px;
      flex-shrink: 0;
    }
    .gauge-text {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      line-height: 1;
    }
    .pct { font-size: 0.78rem; font-weight: 800; }
    .acc-lbl { font-size: 0.55rem; opacity: 0.7; letter-spacing: 0.05em; margin-top: 2px; }
    .foot {
      padding: 8px 14px;
      text-align: center;
      font-size: 0.7rem;
      letter-spacing: 0.04em;
      font-weight: 700;
      color: #fff;
      border-top: 1px solid rgba(194, 155, 81, 0.4);
      background: rgba(0, 0, 0, 0.12);
    }
  `,
})
export class RecentResultComponent {
  private readonly _match = signal<MatchPrediction | null>(null);

  @Input({ required: true })
  set match(value: MatchPrediction) {
    this._match.set(value);
  }
  get match(): MatchPrediction {
    return this._match()!;
  }

  private readonly breakdown = computed<{ outcome: boolean; home: boolean; away: boolean }>(() => {
    const m = this._match();
    if (!m) return { outcome: false, home: false, away: false };
    const predicted = this.split(m.predicted_score);
    const ph = parseInt(predicted.home, 10);
    const pa = parseInt(predicted.away, 10);
    const ah = m.actual_home_goals;
    const aa = m.actual_away_goals;
    if (ah === undefined || aa === undefined || isNaN(ph) || isNaN(pa)) {
      return { outcome: false, home: false, away: false };
    }
    const sign = (a: number, b: number) => (a > b ? 1 : a < b ? -1 : 0);
    return {
      outcome: sign(ph, pa) === sign(ah, aa),
      home: ph === ah,
      away: pa === aa,
    };
  });

  readonly accPct = computed<number>(() => {
    const b = this.breakdown();
    return (b.outcome ? 50 : 0) + (b.home ? 25 : 0) + (b.away ? 25 : 0);
  });

  readonly accFraction = computed<number>(() => this.accPct() / 100);

  readonly gaugeColor = computed<string>(() => {
    const pct = this.accPct();
    const b = this.breakdown();
    if (pct >= 100) return '#43A047';
    if (b.outcome) return '#2E7D32';
    if (pct >= 25) return '#E89B3C';
    return '#C62828';
  });

  readonly actualParts = computed<{ home: string; away: string }>(() => this.split(this._match()?.actual_score));
  readonly predictedParts = computed<{ home: string; away: string }>(() => this.split(this._match()?.predicted_score));

  readonly rationale = computed<string>(() => {
    const m = this._match();
    if (!m) return '';
    const b = this.breakdown();
    if (b.outcome && b.home && b.away) return 'Exact score — perfect call';
    if (b.outcome && (b.home || b.away)) {
      const side = b.home ? m.home_team_name : m.away_team_name;
      return `Correct outcome, ${side} goals exact`;
    }
    if (b.outcome) return 'Correct outcome, scoreline off';
    if (b.home || b.away) {
      const side = b.home ? m.home_team_name : m.away_team_name;
      return `Wrong outcome, but ${side} goals exact`;
    }
    return 'Wrong outcome and scoreline';
  });

  private split(s: string | null | undefined): { home: string; away: string } {
    if (!s) return { home: '-', away: '-' };
    const parts = s.split('-');
    if (parts.length !== 2) return { home: '-', away: '-' };
    return { home: parts[0].trim(), away: parts[1].trim() };
  }
}
