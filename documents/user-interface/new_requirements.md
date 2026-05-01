# Eric FC Website Requirements v1.1

## 1. Project Overview
A streamlined, mobile-first web application for users to track personal football match predictions across major leagues [cite: 5]. The platform emphasizes simplicity, personal history, and strong brand identity [cite: 6].

## 2. Visual Identity & Branding
* **Color Palette**:
    * **Eric FC Navy**: #1B3358 [cite: 9]
    * **Trophy Gold**: #C29B51 [cite: 10]
    * **Shadow Blue**: #4B6A88 [cite: 11]
    * **Stadium White**: #F8F9FA [cite: 12]
* **Design Aesthetic**: Modern, sporty, and minimalist [cite: 13].
* **Typography**: Modern sans-serif (e.g., Inter/Roboto) [cite: 38].
* **Logo**: Prominent use of the "Eric FC" badge [cite: 13].

## 3. General Architecture
* **Single-Page Experience**: The site operates as a continuous scroll with no secondary menus or "Detail View" pages [cite: 17].
* **Database Integration**: All scores and predictions are automatically databased; no "Submit" or "Update" buttons are required.
* **Private Tracker**: No social features; the platform is for private personal use [cite: 39].

## 4. Functional Requirements

### League Selection
* Users toggle between specific leagues to filter content [cite: 20]:
    * 2026 World Cup [cite: 21]
    * Premier League [cite: 22]
    * Ligue 1 [cite: 23]
    * La Liga [cite: 24]
* **UI Update**: Display only one league's content at a time.

### Match Predictions (Active)
* **Inputs**:
    * **Match Result**: Compact selection for Home Win, Draw, or Away Win [cite: 28].
    * **Scoreline**: Exact numerical input for both teams [cite: 29].
* **Feedback**: Results are saved instantly to the database.

### Recent Results & Analytics
* **Comparison**: Side-by-side view of Actual Score vs. User Prediction [cite: 34].
* **Accuracy**: Automated accuracy score (percentage-based) per match [cite: 35].
* **Interaction**: Purely informational; no "Detail View" buttons or external links [cite: 36].

## 5. Platform Specifics

### Mobile Version
* **Layout**: Strictly mobile-first, vertical smartphone layout [cite: 38].
* **Accessibility**: Minimum touch target size of 44x44 pixels for all interactive elements [cite: 38].
* **Navigation**: Simple icon-based league switcher at the top.

### Desktop Version
* **Layout**: Responsive wide layout adapting the single-page mobile experience.
* **Navigation**: Tabs to navigate between "Recent Results" and "Predictions" within the selected league view.
* **Information Density**: Compact match result blocks to utilize screen real estate efficiently.

## 6. Technical Specifications
* **Responsiveness**: Strictly mobile-first optimization [cite: 38].
* **Typography**: Navy text with gold accents on a Stadium White background [cite: 38].
