import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./features/competition-list/competition-list.component').then(
        m => m.CompetitionListComponent,
      ),
  },
  {
    path: 'competition/:id',
    loadComponent: () =>
      import('./features/competition-detail/competition-detail.component').then(
        m => m.CompetitionDetailComponent,
      ),
  },
  {
    path: 'about',
    loadComponent: () =>
      import('./features/about/about.component').then(m => m.AboutComponent),
  },
  { path: '**', redirectTo: '' },
];
