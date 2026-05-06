import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AboutPopoverComponent } from './shared/components/about-popover.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, AboutPopoverComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class AppComponent {
  scrollToTop(): void {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}
