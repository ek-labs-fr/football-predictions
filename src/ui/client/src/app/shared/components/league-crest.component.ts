import { ChangeDetectionStrategy, Component, computed, input, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

const LOCAL_LEAGUE_LOGOS: Record<string, string> = {
  'wc-2026': 'leagues/wc-2026.png',
  'premier-league': 'leagues/premier-league.png',
  'ligue-1': 'leagues/ligue-1.png',
  'la-liga': 'leagues/la-liga.png',
};

@Component({
  selector: 'app-league-crest',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (resolvedSrc() && !failed()) {
      <img
        class="crest-img"
        [src]="resolvedSrc()"
        [alt]="name()"
        (error)="onError()"
      />
    } @else {
      <span class="crest-fallback">{{ fallback() }}</span>
    }
  `,
  styles: `
    :host {
      display: inline-flex;
      width: var(--crest-size, 40px);
      height: var(--crest-size, 40px);
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .crest-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .crest-fallback {
      font-size: 1.6rem;
      line-height: 1;
    }
  `,
})
export class LeagueCrestComponent {
  readonly competitionId = input<string | undefined>(undefined);
  readonly leagueId = input<number | undefined>(undefined);
  readonly name = input<string>('');
  readonly fallback = input<string>('⚽');
  readonly failed = signal(false);

  readonly resolvedSrc = computed<string | null>(() => {
    const compId = this.competitionId();
    if (compId && LOCAL_LEAGUE_LOGOS[compId]) {
      return LOCAL_LEAGUE_LOGOS[compId];
    }
    const id = this.leagueId();
    if (id) {
      return `https://media.api-sports.io/football/leagues/${id}.png`;
    }
    return null;
  });

  onError(): void {
    this.failed.set(true);
  }
}
