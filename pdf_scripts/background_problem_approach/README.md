# Background, problem space, and approach (LaTeX)

TeX sources for the SPF background document:

- `main.tex` — driver (title, abstract, TOC)
- `sections/motivation.tex` — why cheap two-antenna localization
- `sections/problem_space.tex` — task, platforms, hardware, what makes it hard
- `sections/math_phase_to_angle.tex` — **the careful math**: step-by-step from
  array geometry → path difference → time delay → phase difference
  φ = −2π(d/λ)sinθ, phase estimation from IQ, the two ambiguities
  (front–back mirror, d>λ/2 aliasing/grating lobes), sensitivity
  dφ/dθ, the 3-parameter systematic model (c, g, Δθ) used by the dataset
  audit, and circular statistics for the residual noise
- `sections/approach.tex` — physics-shaped inputs + learned mapping,
  two-stage single→paired training, filtering, dataset auditing as method

Build:

```bash
make            # needs pdflatex (sudo apt install texlive-latex-extra)
```

Uses only standard packages (amsmath, booktabs, tikz, hyperref, geometry).
