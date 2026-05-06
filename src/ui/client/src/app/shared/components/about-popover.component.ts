import { ChangeDetectionStrategy, Component, HostListener, signal } from '@angular/core';

@Component({
  selector: 'app-about-popover',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <button
      type="button"
      class="trigger"
      [attr.aria-expanded]="open()"
      aria-haspopup="dialog"
      aria-controls="about-popover-panel"
      (click)="toggle($event)"
    >
      About ERIC FC
    </button>

    @if (open()) {
      <div class="backdrop" (click)="close()" aria-hidden="true"></div>
      <div
        id="about-popover-panel"
        class="panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-popover-title"
        (click)="$event.stopPropagation()"
      >
        <header class="head">
          <h2 id="about-popover-title">About ERIC FC</h2>
          <button type="button" class="close" aria-label="Close" (click)="close()">×</button>
        </header>
        <p>
          ERIC FC is an AI-powered football prediction site. It uses machine learning models
          trained on historical fixture data to forecast scorelines for the FIFA World Cup 2026
          and major European leagues.
        </p>
        <h3>Reference docs</h3>
        <ul class="docs">
          <li>
            <a [href]="docBase + 'technical-architecture.md'" target="_blank" rel="noopener" (click)="close()">
              Technical architecture
            </a>
          </li>
          <li>
            <a [href]="docBase + 'model-card-national.md'" target="_blank" rel="noopener" (click)="close()">
              Model card — National teams
            </a>
          </li>
          <li>
            <a [href]="docBase + 'model-card-clubs.md'" target="_blank" rel="noopener" (click)="close()">
              Model card — Clubs
            </a>
          </li>
        </ul>
      </div>
    }
  `,
  styles: `
    :host { display: inline-block; }

    .trigger {
      background: none;
      border: 0;
      padding: 0;
      font: inherit;
      letter-spacing: inherit;
      text-transform: inherit;
      color: var(--ericfc-shadow-blue);
      cursor: pointer;
      border-bottom: 1px dashed transparent;
      transition: color .15s, border-color .15s;
    }
    .trigger:hover,
    .trigger[aria-expanded="true"] {
      color: var(--ericfc-navy);
      border-bottom-color: var(--ericfc-gold);
    }
    .trigger:focus-visible {
      outline: 2px solid var(--ericfc-gold);
      outline-offset: 3px;
      border-radius: 2px;
    }

    .backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 25, 50, 0.55);
      z-index: 1100;
      animation: fade-in .15s ease;
    }

    .panel {
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: min(520px, calc(100vw - 32px));
      max-height: calc(100vh - 64px);
      overflow-y: auto;
      background: #fff;
      color: var(--ericfc-navy);
      border-radius: 14px;
      box-shadow: 0 24px 60px rgba(15, 25, 50, 0.35);
      padding: 20px 22px 22px;
      z-index: 1101;
      text-align: left;
      letter-spacing: normal;
      text-transform: none;
      font-size: 0.92rem;
      line-height: 1.5;
      animation: fade-in .18s ease;
    }
    .head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 0 12px;
    }
    .panel h2 {
      margin: 0;
      font-size: 1.1rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: var(--ericfc-navy);
    }
    .panel h3 {
      margin: 16px 0 8px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ericfc-shadow-blue);
    }
    .panel p { margin: 0; }

    .close {
      background: none;
      border: 0;
      font-size: 1.5rem;
      line-height: 1;
      color: var(--ericfc-shadow-blue);
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
    }
    .close:hover { background: rgba(15, 25, 50, 0.06); color: var(--ericfc-navy); }
    .close:focus-visible {
      outline: 2px solid var(--ericfc-gold);
      outline-offset: 2px;
    }

    .docs {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .docs a {
      color: var(--ericfc-navy);
      text-decoration: underline;
      text-decoration-color: var(--ericfc-gold);
      text-underline-offset: 3px;
      font-weight: 600;
    }
    .docs a:hover { color: #000; }

    @keyframes fade-in {
      from { opacity: 0; }
      to   { opacity: 1; }
    }

    @media (max-width: 480px) {
      .panel {
        font-size: 0.88rem;
        padding: 16px 18px 18px;
      }
      .panel h2 { font-size: 1rem; }
    }
  `,
})
export class AboutPopoverComponent {
  protected readonly open = signal(false);
  protected readonly docBase =
    'https://github.com/ek-labs-fr/football-predictions/blob/master/documents/';

  protected toggle(event: MouseEvent): void {
    event.stopPropagation();
    this.open.update((v) => !v);
  }

  protected close(): void {
    this.open.set(false);
  }

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    if (this.open()) this.close();
  }
}
