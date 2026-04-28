import { ChangeDetectionStrategy, Component, computed, input, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-team-crest',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (teamId() && !failed()) {
      <img
        class="crest-img"
        [src]="'https://media.api-sports.io/football/teams/' + teamId() + '.png'"
        [alt]="name()"
        (error)="onError()"
      />
    } @else {
      <span class="crest-fallback">{{ initials() }}</span>
    }
  `,
  styles: `
    :host {
      display: inline-flex;
      width: var(--crest-size, 44px);
      height: var(--crest-size, 44px);
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .crest-img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.25));
    }
    .crest-fallback {
      width: 100%;
      height: 100%;
      border-radius: 50%;
      background: #fff;
      color: var(--ericfc-navy);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 0.78rem;
      letter-spacing: 0.04em;
      border: 2px solid var(--ericfc-gold);
    }
  `,
})
export class TeamCrestComponent {
  readonly teamId = input<number | undefined>(undefined);
  readonly name = input<string>('');
  readonly failed = signal(false);

  readonly initials = computed(() => {
    const n = this.name();
    if (!n) return '';
    const words = n.split(/\s+/).filter(Boolean);
    if (words.length === 1) return words[0].slice(0, 3).toUpperCase();
    return (words[0][0] + words[1][0] + (words[2]?.[0] ?? '')).toUpperCase();
  });

  onError(): void {
    this.failed.set(true);
  }
}
