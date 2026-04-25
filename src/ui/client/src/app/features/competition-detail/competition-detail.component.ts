import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { MatTabsModule } from '@angular/material/tabs';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatchCardComponent } from '../../shared/components/match-card.component';
import { PerformanceSummaryComponent } from '../../shared/components/performance-summary.component';
import {
  PastResponse,
  PredictionService,
  UpcomingResponse,
} from '../../services/prediction.service';

@Component({
  selector: 'app-competition-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    MatTabsModule,
    MatProgressSpinnerModule,
    MatButtonModule,
    MatIconModule,
    MatchCardComponent,
    PerformanceSummaryComponent,
  ],
  template: `
    <div class="header">
      <a mat-icon-button routerLink="/" aria-label="Back">
        <mat-icon>arrow_back</mat-icon>
      </a>
      <h2>{{ upcoming?.competition_name || past?.competition_name || 'Competition' }}</h2>
    </div>

    @if (loading) {
      <div class="spinner"><mat-spinner diameter="40" /></div>
    } @else if (error) {
      <p class="error">{{ error }}</p>
    } @else {
      <mat-tab-group>
        <mat-tab [label]="'Upcoming (' + (upcoming?.matches?.length ?? 0) + ')'">
          <div class="tab-content">
            @if (upcoming && upcoming.matches.length > 0) {
              @for (m of upcoming.matches; track m.fixture_id) {
                <app-match-card [match]="m" />
              }
            } @else {
              <p class="empty">No upcoming fixtures.</p>
            }
          </div>
        </mat-tab>

        <mat-tab [label]="'Past (' + (past?.matches?.length ?? 0) + ')'">
          <div class="tab-content">
            @if (past && past.matches.length > 0) {
              <app-performance-summary [perf]="past.performance" [label]="past.label" />
              @for (m of past.matches; track m.fixture_id) {
                <app-match-card [match]="m" />
              }
            } @else {
              <p class="empty">No past results yet.</p>
            }
          </div>
        </mat-tab>
      </mat-tab-group>
    }
  `,
  styles: `
    .header {
      display: flex;
      align-items: center;
      gap: 4px;
      margin-bottom: 16px;
    }
    .header h2 { margin: 0; font-size: 1.3rem; }
    .spinner { display: flex; justify-content: center; padding: 40px; }
    .error { color: #c62828; text-align: center; padding: 20px; }
    .tab-content { padding-top: 16px; }
    .empty { color: rgba(0,0,0,0.5); padding: 20px; text-align: center; }
  `,
})
export class CompetitionDetailComponent implements OnInit {
  upcoming: UpcomingResponse | null = null;
  past: PastResponse | null = null;
  loading = true;
  error: string | null = null;

  constructor(
    private route: ActivatedRoute,
    private predictionService: PredictionService,
  ) {}

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id');
    if (!id) {
      this.error = 'Missing competition id.';
      this.loading = false;
      return;
    }

    let pending = 2;
    const done = () => { pending -= 1; if (pending === 0) this.loading = false; };

    this.predictionService.getUpcoming(id).subscribe({
      next: data => { this.upcoming = data; done(); },
      error: () => { this.error = 'Failed to load upcoming fixtures.'; done(); },
    });
    this.predictionService.getPast(id).subscribe({
      next: data => { this.past = data; done(); },
      error: () => { this.error = 'Failed to load past results.'; done(); },
    });
  }
}
