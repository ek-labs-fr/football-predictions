import { ChangeDetectionStrategy, Component, Input, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatchPrediction, Outcome } from '../../services/prediction.service';
import { TeamCrestComponent } from './team-crest.component';

@Component({
  selector: 'app-upcoming-match',
  standalone: true,
  imports: [CommonModule, TeamCrestComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <article class="card">
      <header class="head">
        <div class="side">
          <app-team-crest [teamId]="match.home_team_id" [name]="match.home_team_name" />
          <span class="name">{{ match.home_team_name }}</span>
        </div>
        <div class="middle">
          <span class="date">{{ match.date | date: 'EEE, MMM d' }}</span>
          <div class="score-row">
            <span class="score-cell">{{ scoreParts().home }}</span>
            <span class="score-sep">–</span>
            <span class="score-cell">{{ scoreParts().away }}</span>
          </div>
        </div>
        <div class="side away">
          <app-team-crest [teamId]="match.away_team_id" [name]="match.away_team_name" />
          <span class="name">{{ match.away_team_name }}</span>
        </div>
      </header>

      <p class="cap">MATCH RESULT</p>
      <div class="pills" role="group" aria-label="Predicted match result">
        <span class="pill" [class.on]="scoreOutcome() === 'home_win'">HOME</span>
        <span class="pill" [class.on]="scoreOutcome() === 'draw'">DRAW</span>
        <span class="pill" [class.on]="scoreOutcome() === 'away_win'">AWAY</span>
      </div>

      @if (match.rationale) {
        <footer class="foot">{{ match.rationale }}</footer>
      }
    </article>
  `,
  styles: `
    .card {
      background: var(--ericfc-navy);
      color: #fff;
      border-radius: 16px;
      padding: 14px 14px 12px;
      box-shadow: 0 6px 18px rgba(27, 51, 88, 0.18);
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: transform .18s ease, box-shadow .18s ease;
      will-change: transform;
    }
    @media (hover: hover) and (min-width: 720px) {
      .card:hover {
        transform: scale(1.02);
        box-shadow:
          0 10px 28px rgba(27, 51, 88, 0.28),
          0 0 0 1px rgba(194, 155, 81, 0.35),
          0 0 22px rgba(194, 155, 81, 0.28);
      }
    }
    .head {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      gap: 8px;
      align-items: center;
    }
    .side {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }
    app-team-crest { --crest-size: 44px; }
    .name {
      font-size: 0.78rem;
      font-weight: 600;
      text-align: center;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .middle {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }
    .date {
      font-size: 0.72rem;
      color: var(--ericfc-gold-300);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .score-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .score-cell {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: #fff;
      color: var(--ericfc-navy);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-family: 'Roboto Condensed', 'Inter', sans-serif;
      font-size: 1.35rem;
      font-weight: 900;
      letter-spacing: -0.02em;
      box-shadow: inset 0 -2px 0 rgba(0, 0, 0, 0.06);
    }
    .score-sep {
      font-weight: 700;
      color: rgba(255, 255, 255, 0.6);
    }
    .cap {
      margin: 4px 0 0;
      font-size: 0.65rem;
      letter-spacing: 0.18em;
      text-align: center;
      color: rgba(255, 255, 255, 0.7);
    }
    .pills {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      background: var(--ericfc-navy-700);
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
    }
    .pill {
      text-align: center;
      padding: 8px 0;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      color: rgba(255, 255, 255, 0.65);
      border-radius: 8px;
      min-height: 36px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }
    .pill.on {
      background: var(--ericfc-gold);
      color: var(--ericfc-navy);
      box-shadow: 0 2px 6px rgba(194, 155, 81, 0.3);
    }
    .foot {
      margin: 4px -14px -12px;
      padding: 8px 14px;
      text-align: center;
      font-size: 0.7rem;
      letter-spacing: 0.04em;
      font-weight: 600;
      color: rgba(255, 255, 255, 0.92);
      border-top: 1px solid rgba(194, 155, 81, 0.4);
      background: rgba(0, 0, 0, 0.12);
    }
  `,
})
export class UpcomingMatchComponent {
  private readonly _match = signal<MatchPrediction | null>(null);

  @Input({ required: true })
  set match(value: MatchPrediction) {
    this._match.set(value);
  }
  get match(): MatchPrediction {
    return this._match()!;
  }

  readonly scoreParts = computed<{ home: string; away: string }>(() => {
    const m = this._match();
    if (!m) return { home: '–', away: '–' };
    const parts = (m.predicted_score || '').split('-');
    if (parts.length !== 2) return { home: '–', away: '–' };
    return { home: parts[0].trim(), away: parts[1].trim() };
  });

  readonly scoreOutcome = computed<Outcome | null>(() => {
    const p = this.scoreParts();
    const h = parseInt(p.home, 10);
    const a = parseInt(p.away, 10);
    if (isNaN(h) || isNaN(a)) return null;
    if (h > a) return 'home_win';
    if (h < a) return 'away_win';
    return 'draw';
  });

  outcomeLabel(o: Outcome): string {
    return o === 'home_win' ? 'HOME' : o === 'away_win' ? 'AWAY' : 'DRAW';
  }
}
