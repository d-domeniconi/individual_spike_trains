# Analysis of Individual Spike Trains — Chapter 3, Paul Miller

Python implementation of the tutorials from **Chapter 3 ("Analysis of Individual Spike Trains")** of Paul Miller's *An Introductory Course in Computational Neuroscience*. The neuron model used throughout is the **Adaptive Exponential Leaky Integrate-and-Fire (AELIF)** model, converted here from MATLAB to Python.

- Tutorial 3.1 — [`tut_3_1.ipynb`](tut_3_1.ipynb): Generating receptive fields with spike-triggered averages
- Tutorial 3.2 — [`tut_3_2.ipynb`](tut_3_2.ipynb): Statistical properties of simulated spike trains
- Simulation driver — [`aelif_sim.ipynb`](aelif_sim.ipynb): runs the AELIF model for each part of the tutorials above and saves the resulting current/spike data

## Topics covered

- AELIF neuron simulation (leaky integrate-and-fire with exponential spike initiation and spike-triggered adaptation)
- Spike-triggered average (STA) and stimulus reconstruction
- Interspike intervals (ISI) and coefficient of variation (CV)
- Fano factor as a function of counting-window size
- Poisson-like spike statistics
- Signal downsampling / binning for spike-train analysis
- NumPy-based neural simulations

## Model and methods

### AELIF neuron model

The membrane potential $V$ and adaptation current $w$ evolve according to the adaptive exponential integrate-and-fire equations:

$$
C \frac{dV}{dt} = G_L\left(E_L - V + \Delta_T \exp\!\left(\frac{V - V_{th}}{\Delta_T}\right)\right) - w + I_{app}
$$

$$
\tau_w \frac{dw}{dt} = a\,(V - E_L) - w
$$

with a spike-and-reset rule: whenever $V$ crosses the spike-detection voltage $V_{max}$, a spike is recorded and

$$
V \rightarrow V_{reset}, \qquad w \rightarrow w + b .
$$

Both state variables are integrated with the explicit (forward) Euler method at a fixed timestep $dt$:

$$
V[j+1] = V[j] + dt \cdot \left.\frac{dV}{dt}\right|_j, \qquad w[j+1] = w[j] + dt \cdot \left.\frac{dw}{dt}\right|_j
$$

| Symbol | Meaning | Units |
|---|---|---|
| $C$ | Membrane capacitance | F |
| $G_L$ | Leak conductance | S |
| $E_L$ | Leak (resting) potential | V |
| $V_{th}$ | Threshold potential | V |
| $\Delta_T$ | Threshold slope factor | V |
| $V_{reset}$ | Post-spike reset potential | V |
| $V_{max}$ | Spike-detection/cropping voltage | V |
| $\tau_w$ | Adaptation time constant | s |
| $a$ | Sub-threshold adaptation conductance | S |
| $b$ | Spike-triggered adaptation increment | A |
| $I_{app}$ | Applied (input) current | A |

#### Parameter sets

**Tutorial 3.1** — single neuron, piecewise-constant stimulus ($dt = 0.02$ ms, $T = 200$ s):

| $G_L$ | $C$ | $E_L$ | $V_{th}$ | $V_{reset}$ | $\Delta_T$ | $\tau_w$ | $a$ | $b$ | $V_{max}$ |
|---|---|---|---|---|---|---|---|---|---|
| 8 nS | 100 pF | −60 mV | −50 mV | −80 mV | 2 mV | 50 ms | 10 nS | 1 nA | 50 mV |

**Tutorial 3.2, parts a/b** — Gaussian-noise-driven neuron ($dt = 0.1$ ms, $T = 100$ s; $G_L=10$ nS, $C=100$ pF, $E_L=-70$ mV, $V_{th}=-50$ mV, $V_{reset}=-80$ mV, $\Delta_T=2$ mV, $\tau_w=150$ ms, $V_{max}=50$ mV):

| Part | $a$ | $b$ | Noise $\sigma$ |
|---|---|---|---|
| a — no adaptation | 2 nS | 0 nA | 50 pA·s$^{1/2}$ |
| b — with adaptation | 2 nS | 1 nA | 50 pA·s$^{1/2}$ |

**Tutorial 3.2, part c** — reduced noise, no adaptation ($a = 2$ nS, $b = 0$ nA, other cell parameters as above), with three DC current offsets:

| Case | Noise $\sigma$ | DC offset |
|---|---|---|
| c0 | 20 pA·s$^{1/2}$ | 0 |
| c1 | 20 pA·s$^{1/2}$ | +0.1 nA |
| c2 | 20 pA·s$^{1/2}$ | +0.2 nA |

#### Input current

- **Piecewise-constant stimulus (Tutorial 3.1):** 40,000 values drawn i.i.d. from $\mathcal{U}(-0.5\text{ nA}, +0.5\text{ nA})$, each held fixed for a 5 ms block (built by `time_and_current_vectors`), giving a 200 s stimulus at a 0.02 ms simulation timestep.
- **Gaussian white-noise current (Tutorial 3.2):** at every timestep, $I_{app}$ is drawn from $\mathcal{N}(0,\ \sigma/\sqrt{dt})$, so the noise power is independent of $dt$; optionally shifted by a constant DC offset (parts c1/c2).

### Spike-train metrics

**Spike-triggered average (STA).** For $N$ spikes at times $\{t_i\}$, the STA is the average stimulus waveform around each spike:

$$
\mathrm{STA}(\tau) = \frac{1}{N}\sum_{i=1}^{N} I_{app}(t_i + \tau), \qquad \tau \in [-t_{minus},\, t_{plus}]
$$

implemented in `STA()`; spike windows that would extend past the edges of the recording are discarded. Tutorial 3.1 uses $t_{minus} = 75$ ms and $t_{plus} = 25$ ms, and plots the result against $-\tau$ so that positive lag represents the causal effect of the stimulus on the spike. Before computing the STA, the stimulus and spike train are downsampled from the simulation timestep (0.02 ms) to a 1 ms analysis bin using `expandbin` (bin-averaging) and `downsample_spikes` (bin-average, then re-threshold to binary).

**Interspike interval (ISI) and coefficient of variation (CV).** Spike times are read off the indices where the spike train equals 1; ISIs are the differences between consecutive spike times, $\mathrm{ISI}_k = t_{k+1}-t_k$. The CV summarizes firing irregularity:

$$
CV = \frac{\sigma_{ISI}}{\mu_{ISI}}
$$

$CV \approx 1$ is characteristic of a Poisson process (exponentially distributed ISIs); $CV < 1$ indicates more regular firing, as expected when strong spike-triggered adaptation is present.

**Fano factor.** Spikes are counted in consecutive, non-overlapping windows of fixed size; the Fano factor is the variance-to-mean ratio of the resulting spike counts:

$$
F = \frac{\mathrm{Var}(n)}{\langle n \rangle}
$$

$F = 1$ for a homogeneous Poisson process. The notebooks compute $F$ for a single 100 ms window (`fano_factor`) and then sweep the window size from 10 ms to 1 s (`fano_factor_per_window`) to see how $F$ depends on the counting timescale, particularly its sensitivity to spike-triggered adaptation.

## Repository structure

```markdown
.
├── aelif_sim.ipynb       # Runs the AELIF model for every tutorial part/seed and saves data/*.npz
├── tut_3_1.ipynb         # Tutorial 3.1: spike-triggered average of a time-varying stimulus
├── tut_3_2.ipynb         # Tutorial 3.2: ISI, CV, and Fano factor of simulated spike trains
├── modules/
│   └── STA.py            # Shared helper functions imported by the tutorial notebooks
├── data/                 # Saved simulation outputs (.npz), organized per tutorial
├── figs/                 # Figures exported from the notebooks
├── requirements.txt
├── LICENSE
└── README.md
```

## The `modules/STA.py` module

Helper functions shared by the tutorial notebooks:

| Function | Description |
|---|---|
| `time_and_current_vectors(T, d_t, V, interval=5e-3)` | Builds the simulation time vector and the applied-current vector, holding each value of `V` constant for a fixed interval (5 ms by default). |
| `expandbin(old_vector, old_dt, new_dt)` | Downsamples a time series by averaging consecutive bins, used to reduce the simulation timestep to a coarser analysis timestep. |
| `downsample_spikes(spikes, old_dt, new_dt)` | Downsamples a binary spike train via `expandbin`, then re-thresholds the result back to a binary (0/1) train. |
| `STA(applied_current_vector, spikes, dt, t_minus=75, t_plus=25)` | Computes the spike-triggered average of the applied current in a window `[-t_minus, +t_plus]` around every spike. |

## Installation

### Clone the repository

```bash
git clone https://github.com/d-domeniconi/tutorials.git
cd tutorials
```

### Create the virtual environment named `tut`

```bash
python3 -m venv tut
```

### Activate the environment

On macOS/Linux:

```bash
source tut/bin/activate
```

On Windows (Command Prompt):

```bash
tut\Scripts\activate.bat
```

On Windows (PowerShell):

```powershell
.\tut\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

1. Run `aelif_sim.ipynb` first to simulate the AELIF neuron and generate the `.npz` data files used by the tutorials (each part/seed is saved separately under `data/`).
2. Run `tut_3_1.ipynb` to reproduce the spike-triggered average analysis.
3. Run `tut_3_2.ipynb` to reproduce the ISI, CV, and Fano-factor analysis.

Figures generated by the notebooks are saved to `figs/`.

## Reference

Miller, P. (2018). *An Introductory Course in Computational Neuroscience*. MIT Press. — Chapter 3, "Analysis of Individual Spike Trains."

## License

This project is licensed under the [MIT License](LICENSE).
