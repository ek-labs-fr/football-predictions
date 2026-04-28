import { ChangeDetectionStrategy, Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Competition } from '../../services/prediction.service';
import { LeagueCrestComponent } from './league-crest.component';

@Component({
  selector: 'app-league-switcher',
  standalone: true,
  imports: [CommonModule, LeagueCrestComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <span class="cap">SELECT LEAGUE</span>
    <div class="row">
      @for (c of orderedCompetitions(); track c.id) {
        <button
          type="button"
          class="card"
          [class.selected]="c.id === selectedId"
          (click)="onClick(c.id)"
          [attr.aria-pressed]="c.id === selectedId"
        >
          <app-league-crest
            class="crest"
            [competitionId]="c.id"
            [leagueId]="c.league_id"
            [name]="c.name"
            [fallback]="crestFor(c.id)"
          />
          <span class="name">{{ shortName(c) }}</span>
        </button>
      }
    </div>
  `,
  styles: `
    :host {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      width: 100%;
    }
    .cap {
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      color: #fff;
      opacity: 0.9;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(4, minmax(64px, 1fr));
      gap: 10px;
      width: 100%;
    }
    .card {
      position: relative;
      min-height: 92px;
      min-width: 44px;
      padding: 10px 6px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      background: var(--ericfc-navy-700);
      color: #fff;
      border: 2px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      font-family: inherit;
      cursor: pointer;
      overflow: hidden;
      transition: border-color .15s, transform .1s, box-shadow .15s, background .15s;
    }
    .card:hover { border-color: rgba(194, 155, 81, 0.55); }
    .card.selected {
      background: linear-gradient(180deg, rgba(194, 155, 81, 0.16), rgba(194, 155, 81, 0.04) 60%, var(--ericfc-navy-700));
      border-color: var(--ericfc-gold);
      box-shadow:
        0 0 0 2px rgba(194, 155, 81, 0.35),
        0 0 18px 2px rgba(194, 155, 81, 0.45),
        0 6px 18px rgba(0, 0, 0, .25);
    }
    .crest {
      --crest-size: 44px;
    }
    .name {
      font-size: 0.72rem;
      font-weight: 600;
      text-align: center;
      line-height: 1.15;
    }
    @media (min-width: 900px) {
      :host { width: auto; gap: 8px; align-items: flex-end; }
      .row { grid-template-columns: repeat(4, 120px); }
      .card { min-height: 104px; padding: 12px 10px; }
      .crest { --crest-size: 52px; }
      .name { font-size: 0.78rem; }
    }
  `,
})
export class LeagueSwitcherComponent {
  @Input({ required: true }) competitions: Competition[] = [];
  @Input() selectedId: string | null = null;
  @Output() select = new EventEmitter<string>();

  onClick(id: string): void {
    if (id !== this.selectedId && typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
      navigator.vibrate(8);
    }
    this.select.emit(id);
  }

  private readonly order = ['premier-league', 'la-liga', 'ligue-1', 'wc-2026'];

  orderedCompetitions(): Competition[] {
    const byId = new Map(this.competitions.map(c => [c.id, c]));
    return this.order.map(id => byId.get(id)).filter((c): c is Competition => !!c);
  }

  shortName(c: Competition): string {
    if (c.id === 'wc-2026') return '2026 World Cup';
    return c.name;
  }

  crestFor(id: string): string {
    switch (id) {
      case 'wc-2026': return '🏆';
      case 'premier-league': return '🦁';
      case 'ligue-1': return '🇫🇷';
      case 'la-liga': return '🇪🇸';
      default: return '⚽';
    }
  }
}
