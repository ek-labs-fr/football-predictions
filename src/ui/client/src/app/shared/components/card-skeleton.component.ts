import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

type SkeletonVariant = 'upcoming' | 'recent';

@Component({
  selector: 'app-card-skeleton',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <article class="skel" [class.upcoming]="variant === 'upcoming'" [class.recent]="variant === 'recent'" aria-hidden="true">
      @if (variant === 'upcoming') {
        <div class="head">
          <div class="side">
            <div class="bar crest"></div>
            <div class="bar name"></div>
          </div>
          <div class="middle">
            <div class="bar date"></div>
            <div class="score-row">
              <div class="bar score-cell"></div>
              <div class="bar score-cell"></div>
            </div>
          </div>
          <div class="side">
            <div class="bar crest"></div>
            <div class="bar name"></div>
          </div>
        </div>
        <div class="bar pills"></div>
      } @else {
        <div class="bar head-bar"></div>
        <div class="body">
          <div class="left">
            <div class="bar row"></div>
            <div class="bar row"></div>
          </div>
          <div class="bar gauge"></div>
        </div>
        <div class="bar foot"></div>
      }
    </article>
  `,
  styles: `
    :host { display: block; }
    .skel {
      background: var(--ericfc-navy);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 6px 18px rgba(27, 51, 88, 0.18);
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 158px;
    }
    .skel.recent {
      padding: 0;
      overflow: hidden;
      border: 1px solid rgba(194, 155, 81, 0.25);
    }
    .bar {
      background: linear-gradient(
        90deg,
        rgba(255, 255, 255, 0.06) 0%,
        rgba(255, 255, 255, 0.16) 50%,
        rgba(255, 255, 255, 0.06) 100%
      );
      background-size: 200% 100%;
      border-radius: 6px;
      animation: shimmer 1.4s ease-in-out infinite;
    }
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
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
    }
    .crest { width: 44px; height: 44px; border-radius: 50%; }
    .name { width: 70%; height: 12px; }
    .middle { display: flex; flex-direction: column; align-items: center; gap: 6px; }
    .date { width: 48px; height: 10px; }
    .score-row { display: flex; gap: 6px; }
    .score-cell { width: 36px; height: 36px; border-radius: 8px; }
    .pills { width: 100%; height: 44px; border-radius: 10px; }

    .recent .head-bar { width: 100%; height: 36px; border-radius: 0; }
    .recent .body { display: flex; gap: 14px; padding: 12px 14px; align-items: center; }
    .recent .left { flex: 1; display: flex; flex-direction: column; gap: 6px; }
    .recent .row { width: 80%; height: 14px; }
    .recent .gauge { width: 64px; height: 64px; border-radius: 50%; flex-shrink: 0; }
    .recent .foot { width: 100%; height: 26px; border-radius: 0; }
  `,
})
export class CardSkeletonComponent {
  @Input() variant: SkeletonVariant = 'upcoming';
}
