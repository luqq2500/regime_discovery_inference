# Regime Discovery — Latent Traversal Analysis UI

Interactive desktop UI for exploring latent market regimes via the `LatentTraversalAnalysisUseCase`.

## Architecture (MVVM)

```
RegimeDiscovery/
├── core/                     # UNTOUCHED business logic
│   ├── domain.py
│   ├── repository/
│   ├── service/
│   └── usecase/
├── view/                     # NEW UI layer only
│   ├── styles.py             # Dark theme (one vibrant blue)
│   ├── viewmodels/           # State + UseCase orchestration
│   │   ├── analysis_vm.py
│   │   └── repository_vm.py
│   └── views/                # Pure widgets
│       ├── main_window.py
│       ├── config_panel.py
│       ├── traversal_plot.py
│       ├── diagnostics_table.py
│       ├── comparison_view.py
│       └── embeddings_view.py
├── run_ui.py                 # Entry point + DI wiring
└── requirements.txt
```

**Rules followed**
- Core module never modified.
- View code depends only on `core.usecase` + `core.repository` (plus services only inside the single entry-point for DI).
- Strict SRP: ViewModels own state & requests; Views only render & forward events.

## Features

| Feature | Description |
|---------|-------------|
| **Regime Selector** | Multi-select dimensions 1–9 with live DOF |
| **σ Range** | Dual spinboxes for market-condition sweep |
| **Top-N / SNR Filter** | Controls the stats filter used by the usecase |
| **Sunburst Overview** | Hierarchical Regime → Asset → Horizon impact (first tab) |
| **Traversal Curves** | Interactive pyqtgraph lines of asset % change vs σ |
| **Diagnostics Table** | Filtered / full mean·std·SNR tables + CSV export |
| **Regime Comparison** | Overlay multiple regimes on the same asset |
| **Repository Explorer** | PCA / t-SNE / UMAP of embeddings (n_components=2..9) + 2D/3D viz + model params + DOFs |

## Styling

- Dark mode (`#0f0f0f` background)
- Single accent: vibrant blue `#00b4ff`
- Minimalist cards, clean typography, subtle borders

## Quick Start

```bash
# From project root
pip install -r requirements.txt

# Make sure the .pt model files are present:
#   core/service/encoder.pt
#   core/service/decoder.pt
#   core/service/scaler.pt
#   core/repository/repository.pt

python run_ui.py
```

## Usage Flow

1. Select one or more **Regimes** on the left.
2. Adjust **σ Min / Max** (default −3 … +3).
3. Set **Top N** and **SNR threshold**.
4. Click **Run Analysis**.
5. Explore the four tabs:
   - Traversal Curves
   - Diagnostics
   - Regime Comparison
   - Repository Explorer

## Notes

- Raw % values are log-return inverse-transformed percentages.
- DOFs come from the repository (Student-t style latent).
- All computation stays inside the existing UseCase; the UI is a pure consumer.
