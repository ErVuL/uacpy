# Broadband products: the reasoning behind the results methods

The docstrings of `Field`, `Arrivals` and `Modes` state what each method takes, returns and refuses. The
reasoning behind them lives here: why a quantity is defined the way it is, the literature it follows, and the
measurements that set its rules. Each section belongs to one method, whose docstring points here.

## Cutting a transfer function to a pulse length (`Field.truncate_response`)

**Why a transfer function has a pulse length in it at all.** `H(f)`
as a model returns it is the continuous-wave answer: every path
present at once, interfering. A pulse of duration `T` does not meet
that channel. Two copies of it interfere only where they overlap, so
arrivals further apart than `T` land as separate, non-interfering
echoes — "we choose individual arrivals and measure their travel
times, amplitudes, and waveforms **when the signals are separable in
the time domain**. If the multiple arrivals are not separable, both
the phases and amplitudes of the components determine how they
interfere" (Medwin and Clay, *Fundamentals of Acoustical
Oceanography*, sect. 3.4.5, "Sum of multiple arrivals"). Jensen et
al. give the test as an operation rather than a rule: "filter these
results within a specified bandwidth in order to obtain the pulse
structure that indicates whether the arrivals are actually separated
in time" (*Computational Ocean Acoustics*, sect. 2.4.4.1, pointing on
to sect. 8.3.1). Ainslie names multipath first among the causes of
coherence loss and prices two replicas in Table 6.9: the input
carries `a^2 + b^2` and the matched filter's output only `a^2`,
worst case 3 dB for equal amplitudes (*Sonar Performance Modeling*,
sect. 6.2.6).

**The recipe is not new.** Transforming a *windowed* impulse response
is standard practice, and named: "time windows can also be used in
separating various components of a transient signal from each other
('gating'), say, separating an impulse that is due to a direct sound
wave from other pulses that are due to reflected waves" (Jacobsen and
Juhl, *Fundamentals of General Linear Acoustics*, sect. B.3.2 "Time
Windows"). Room acoustics does this exact thing and calls the result
the **short-term spectrum** — "a Fourier transform of the first
64 msec of the impulse response after the direct sound has arrived
... windowed using a quarter period cosine squared window ... The
windowing is necessary to prevent the sudden cutoff of the impulse
producing spurious effects in the spectrum", with the length set by
"the integration time of the ear" (Everest and Pohlmann, *Master
Handbook of Acoustics*, "Prediction of room response"). Everything
here is that, with the pulse length in place of the ear's
integration time, and `window='hann'` in place of the cosine
squared and for the reason they give.

The equivalence to smoothing `H` is the convolution theorem, and
the taper's price is its own main lobe: "applying a window `w[n]`
to a signal `x[n]` is the same as convolving the Fourier transform
of the window `W` with the signal's Fourier transform `X` ...
de-emphasizing the data near the window edges has the effect of
shortening the RMS duration and therefore broadening the RMS
bandwidth" (Abraham, sect. 4.10 "Windowing and window functions") —
which is why `'hann'` removes `None`'s skirt and attenuates a
path part-way out instead. And the record must hold the response
before any of this means anything: it "must be selected large enough
that it contains the entire transient response at each receiver so as
to eliminate the aliasing" (Jensen et al., sect. 8.2.1.3 "Time
windowing and sampling"), which is what the wrap-around warning of
`Field.truncate_response` measures.

**How this relates to propagation loss.** Abraham defines the
propagation loss of a pulse twice over (sect. 3.2.4.2 "Propagation
loss and the channel frequency response"), and the two definitions
part company exactly here. In time, `L_p` is the source's
mean-square over the pulse divided by the received mean-square over a
window of the SAME duration `T` — energy landing outside the window
is not counted. In frequency, by Parseval, `L_p = int |U_o|^2 df /
int |H U_o|^2 df`, which runs over ALL time and counts every path.
Parseval needs the whole signal, so the two agree only while the
received pulse fits in the window.

Measured on two paths of amplitude 1 and 0.7, the second `gap`
behind the first at 0.1 s, over 25 Hz-4 kHz at `df` = 1 Hz, with a
rectangular 20-cycle 1 kHz burst (`tone_burst(1000, 20, 48000,
window=None)`, `T` = 20 ms) as the weight. Each column is the
broadband propagation loss in dB (negative where the second path
adds): the frequency form is `broadband_loss`; the gate is the
received energy in `[0.1, 0.1 + T]` s against the pulse's own; the
last two are `broadband_loss` after `truncate_response(T,
origin=0.1, window=…)`, centred on the first path.

```text
=======  ==========  ======  ======  ======
gap      freq form   gate    none    hann
=======  ==========  ======  ======  ======
5 ms     -4.048      -3.831  -4.049  -3.527
10 ms    -3.405      -2.887  -3.407  -1.680
15 ms    -2.648      -1.679  -2.650  -0.260
30 ms    -1.732      +0.002  +0.001  +0.001
=======  ==========  ======  ======  ======
```

`window=None` (rectangular) reproduces the frequency form while
the copies can overlap, and the gate once they cannot — it is the
criterion the other two only bracket. The frequency form never
discards: at 30 ms it still adds the late path's energy, and
`10*log10(1 + 0.7^2)` = 1.732 dB is the whole of its -1.732. It
cannot tell "interferes" from "arrives separately", because averaging
the fringe away under `|U_o|^2` leaves the energy behind. The
`T`-wide gate discards from about half a duration out, being `T`
wide where overlap needs `2T`.

`'hann'` costs 0.5 dB on a path a quarter of the way out, 1.7 dB
half way and 2.4 dB at three quarters, and nothing at all once a path
is beyond. On a channel whose paths sit part-way out, prefer the
rectangle and pay its skirt.

**Which duration.** Ainslie's table is for a separation *small*
against the transmitted pulse `T` and *large* against the
compressed one `1/B`, so what a receiver resolves is `1/B` and
not `T`. For a pulse with `BT` near 1 — a plain tone burst, an
unshaped symbol — the two coincide and `duration = T` is right.
For a chirp or any waveform the receiver compresses, `BT >> 1` and
the separable unit is `1/B`: pass that instead, or this removes
paths the receiver would still have resolved.

In the frequency domain it is the same statement: a signal of
duration `T` has `1/T` of spectral resolution, so structure in
`H` finer than `1/T` — which is what an arrival `T` or more
late puts there — is not something it can see. This method removes
exactly that structure.

**What it is not.** With paths in hand the exact receiver-side answer
is `uacpy.core.results.Arrivals.channel_taps`, which applies
the receiver's own pulse at its decision instants and returns the
far echoes as separate taps rather than discarding them. This is the
version for a model that has no paths — a wave model returns a field,
and the only way to ask it which arrivals are separable is to look at
its response.

**Two copies overlap when their delays differ by less than the pulse
length**, so the window reaches `duration` EITHER SIDE of the
origin — it is `2 * duration` wide. A path further out than that
cannot overlap the one at the origin however they are aligned.

**The band sets a floor on this.** A band `B` wide localises a path
no better than `1/B`, so each arrival appears in the response as a
kernel that wide with skirts around it, and a window cannot separate
two arrivals closer than that however short it is. That is the
temporal resolution cell, not a defect of the cut: refine the band,
not the window.

## Sound exposure level and the record length (`Field.sound_exposure_level`)

A pulse's currency is energy, not mean-square pressure. Abraham
introduces it for exactly this population — "many acoustic signals
are short duration and have varying amplitudes (e.g., a marine
mammal acoustic emissions, an active sonar echo, or a communications
packet). In practice such signals are called transient signals;
however, in a mathematical sense they are energy signals because
their total energy is finite" — and defines the energy flux density
as `(1/rho c) int p^2 dt` (*Underwater Acoustic Signal
Processing*, sect. 3.2.1.5). This returns the time integral itself,
in dB, which is **sound exposure level** as ISO 18405 defines it and
as the marine-mammal exposure criteria are written in (Southall et
al. 2019, where it is the weighted metric paired with peak sound
pressure level). Ainslie builds the active sonar equation on the
same integral, as the energy propagation factor behind total path
loss and energy source level (*Sonar Performance Modeling*,
sect. 3.3.2.1).

**A fold corrupts this, and neither method can tell you.**
Sampling `H(f)` every `df` periodises the impulse response at
`1/df`, so an arrival later than that lands back on the early part
and adds **coherently** to what is there. Parseval preserves the
energy of the *aliased* record, which is not the energy of the true
response, and the cross term is signed: measured over 400 wrapped
delays on a two-path channel the error ran from **-9.27 dB to
+2.75 dB**, exceeding half a decibel in 42.8 % of cases, with no
warning in any of them. `broadband_loss` reads the same
undersampled `H` and carries the identical bias, so
`SEL = ESL - TPL` still closes while both sides are wrong
together — that identity pins the two routes' consistency, not
either one's correctness.

This is Jensen et al.'s aliasing term: the time window "must be
selected large enough that it contains the entire transient response
at each receiver so as to eliminate the aliasing", and the duration
"is not only controlled by the source signal, but also by the
dispersive nature of the waveguide" (*Computational Ocean
Acoustics*, sect. 8.2.1.3) — which is why a check against the pulse
length cannot stand in for one against the channel.

It cannot be detected from `H(f)`: a folded arrival sitting on top
of the direct one is signature-identical to a clean single arrival.
The grid has to be sized before the run, from the arrivals, which is
what `uacpy.core.results.Arrivals.synthesis_band` is for.

**The record must beat twice the delay spread, not once.** Keep
`dtau * df < 0.5` for a path pair `dtau` apart — a record longer
than `2 * dtau` — not merely long enough to hold the arrivals.
Validated on two band configurations: inside the rule the error
stayed within 0.034 dB over 25 Hz-4 kHz and 0.111 dB over
900-1100 Hz, against 0.81 / 2.86 dB beyond it and -9.29 dB at the
worst point found. It bounds the error rather than removing it; on a
narrow band a tenth of a decibel survives.

The size of the error beyond the rule is **not** a function of
`dtau * df` alone — it also turns on `frac(f0 * dtau)`, the
fringe's phase at the band's first sample. Holding the product at
1.5 and moving only the band start gave 0.000002, 0.049917 and
0.028191 dB for `f0` = 25, 40 and 55 Hz. So no spot check settles
it: a single delay on a single axis can land anywhere from a null to
the maximum.

## Broadband propagation loss (`Field.broadband_loss`)

**Why not just run an incoherent model.** Because that is the
approximation, not the quantity. The KRAKEN manual offers it as one
— "if one is comparing to measured data which has been taken by
averaging over frequency one can often simulate the resulting
smoothed result by an incoherent TL" — and Ainslie's Eq. 11.47 is
the same step, dropping the relative phase of the ray arrivals. He
marks where it fails: the step "neglects coherent interference
effects such as cancellation between direct and surface-reflected
paths. This coherent effect is not negligible, even for incoherent
broadband processing, if the distance between the sonar (or target)
and the sea surface is a few wavelengths or less at the center
frequency, in which case use of Equation (11.46) is required."
A Lloyd mirror is exactly that case. This is Eq. 11.46, so it keeps
the interference the band is too narrow to wash out and averages
away only what the band can reach — and it runs on any model that
returns `H(f)`, where `RunMode.INCOHERENT_TL` is Bellhop's and
Kraken's alone.

Jensen's **semicoherent** loss (sect. 3.3.5.4) is a third thing
again: a shading function applied to an incoherent sum, which he
introduces as one of a "variety of techniques" that "all tend to be
somewhat informal and partially empirically based". Prefer this
where the band is known.

**What it does not do** is gate. Averaging over the band removes the
*interference* between paths further apart than `1/B`, and leaves
their energy in the sum — Ainslie's energy definition takes "time
intervals chosen to contain the whole of the transmitted pulse"
(sect. 3.3.2.1). Discarding a late path instead is
`truncate_response`, which answers a receiver-side question
about one cell, not a propagation quantity over a grid.

## From a trace back to H(f) (`Field.to_transfer_function`)

The final rotation is what makes it an inverse rather than merely a
spectrum. A record starting at `t0` carries that offset in every
sample, so a bare `rfft` returns `H` multiplied by
`exp(+2 pi i f t0)` — correct in magnitude and wrong in phase,
which is invisible until something interferes two of them. Verified
against a two-path `H`: with the rotation the round trip
reproduces it to `max|err| = 0.0000`; without it, 3.16.

**Exact on the grid the trace came from.** The inverse returns the
rfft bins `k·Δf` of the record. When the synthesised band started
on a multiple of `Δf` those ARE the model's frequencies and the
round trip closes to rounding (4.7e-16 relative, measured on a
two-path `H` over 25-999 Hz at `Δf = 1` Hz). When it did not —
`to_time_trace` places such a band with a common bin offset — the
spectrum comes back on the integer-`Δf` grid instead, so its
frequencies are not the model's, and the offset band's non-integer
cycles per record leak: measured on the same `H` started at
25.3 Hz, 2.5e-3 relative inside the band and 3.1e-2 at its edge bin.
Compare such a round trip after interpolating, or build the model
grid on multiples of `Δf`.

**The band is restricted, not extended.** An `rfft` of an
`N`-sample record returns bins from 0 to the Nyquist frequency,
but a trace synthesised from a 100-995 Hz field supports nothing
outside that — the other bins are the synthesis's own edges, and
returning them would invent data. The band comes from the Field's
identity when it has one (every trace this package synthesises
does), or from `band`.

**Compared with** `extract_tone`, the careful
single-frequency answer evaluated AT the frequency rather than at
the nearest bin, this is the broadband carrier-level one.

## Peak sound pressure level (`Field.peak_sound_pressure_level`)

**It is not recoverable from any band average.** A peak is a
property of the waveform in time, so no reduction of `|H(f)|`
yields it — which is why the criteria name both metrics rather than
one, and why this needs the synthesis that
`broadband_loss` does not.

**Unlike SEL, it depends on the output sample rate.** A maximum is
a sample, not an integral, so a coarse grid can miss the true crest
between samples. Measured on a two-path channel with a 5-cycle
500 Hz burst, sweeping `nfft` from 320 to 65536 (8 kHz to 1.6 MHz
of output rate): the peak moved **0.0615 dB** and the SEL of the
same traces moved **0.0000 dB**, Parseval holding it exactly. The
peak is converged by about `nfft` = 4096; the automatic size lands
within 0.004 dB of that here. Raise `nfft` if a fraction of a
decibel matters, and note the cost is only in the transform length.

## The synthesis band of an arrival set (`Arrivals.synthesis_band`)

The window is the primitive here and the spacing follows from it, the
way the textbook formulation puts it: "it is convenient to properly
select the time windowing T and sampling dt needed to represent the
response at all the receivers. This, in turn, constrains the
frequency sampling" (Jensen, Kuperman, Porter and Schmidt,
*Computational Ocean Acoustics*, sect. 8.2). Pass `record` to state
that window. Leave it out and it is derived from the arrivals: the
span holding `energy_fraction` of their energy
(`energy_support`), times `margin`.

Folding is a property of the FREQUENCY route, not of the model.
Sampling `H(f)` every `df` and inverse-transforming reproduces the
true response repeated every `1/df`, so whatever does not fit lands
back at the wrong time. A ray model does not have to take that route:
"the ray/beam process calculates the amplitudes and travel-times of
all the echoes and can therefore calculate the received timeseries by
simply summing up the echoes" (Bellhop User Guide sect. 9). That is
`RunMode.TIME_SERIES` (`to_time_series`),
where an echo past the window is omitted rather than folded — the
honest truncation, at the cost of giving up the transfer function.

There is a third way, which keeps the whole arrival set on the
frequency route and makes the wrap-around harmless instead: displace the
frequency contour to `w + i*delta`, which damps the synthesised
trace by `exp(-delta*t)`, so energy that wraps a full record length
returns `exp(-delta*T)` down and the damping is undone on the trace
afterwards. Mallick and Frazer put `delta = log(50)/T` — a factor of
50 — and warn against more, which invents arrivals; the vendored
OASES does exactly that (`third_party/oases/src/unoasp22.f`,
`OMEGIM`). It is not what this does, because it needs the transfer
function evaluated at COMPLEX frequency and Bellhop takes a real one.
Note also that the contour MAGNIFIES aliasing from earlier windows,
so it additionally requires the record to start before the first
arrival.

## H(f) of an arrival set (`Arrivals.transfer_function`)

**Why this exists when `RunMode.BROADBAND` already does it.** That
run mode answers for every path the model found. This answers for the
paths left after `filter`, `window`,
`filter_by_bounces` — and *which paths belong in the sum* is a
question about the SIGNAL, not about the channel. Two arrivals
interfere only if the transmitted waveform is long enough for their
copies to overlap at the receiver: Medwin and Clay put it as
"we choose individual arrivals and measure their travel times,
amplitudes, and waveforms **when the signals are separable in the
time domain**. If the multiple arrivals are not separable, both the
phases and amplitudes of the components determine how they interfere"
(*Fundamentals of Acoustical Oceanography*, sect. 3.4.5), and Jensen
et al. prescribe the test — "filter these results within a specified
bandwidth in order to obtain the pulse structure that indicates
whether the arrivals are actually separated in time"
(*Computational Ocean Acoustics*, sect. 2.4.4.1).

So for a pulse of duration `T`, the transfer function that governs
what one copy of it becomes is `Arrivals.transfer_function` on
`arrivals.window(delay=(t0, t0 + T))`; paths outside that window
arrive as separate, non-interfering echoes and belong in a tap list
(`channel_taps`), not in the sum. Summing them anyway is the
continuous-wave answer, which is a different measurement.

## The modal-sum prefactor (`Modes.modal_pressure_field`)

The textbook prefactor `i·exp(−iπ/4)/(ρ_s·√(8πr))` is written for
`e^{−iωt}` with outgoing `e^{+i k r}`; conjugating it into AT's
`e^{i(ωt − k r)}` and folding in TL's free-field 1 m reference
`1/(4π)` gives the `exp(−iπ/4)·√(2π/r)/ρ_s` used here
(`4π/√(8π) = √(2π)`). That `√(2π)` is the magnitude of AT's own
modal-evaluator prefactor `i·√(2π)·exp(iπ/4)`
(`KrakenField/EvaluateMod.f90:34`), so `|P|` — and therefore TL
— lands on the same absolute scale as `field.exe`; the two differ
by an overall `−1`, which neither `−20·log10|P|` nor a phase
*difference* across the grid can see.

## The finite-difference group velocity (`Modes.group_velocity_between`)

which moves with frequency and mode order: on a 100 m Pekeris guide
(c 1500/1800 m/s, ρ 1.0/1.8) at 100 Hz the optimum is Δf = 1 Hz, at
7.3e-06 relative against exact roots; a decade below it the answer is
6x worse, two decades below 85x worse, and at Δf = 1e-4 Hz the seven
modes collapse onto four distinct speeds. Upcasting `k` recovers
none of this — the bits were never written — so a step whose storage
floor exceeds 1e-5 is warned about instead.

The warning does **not** prescribe a step. `|d²k_r/dω²|` is not
visible from two frequencies, and a step chosen from the floor alone
made the answer worse in 14 of 30 firings on five ideal waveguides, by
up to 16x (measured). It gives the test instead: recompute at twice
the separation and keep the wider answer only if v_g stays inside the
floor. `uacpy.acoustic_signal.modal_group_velocity`, which is
handed a whole sweep, runs that test itself and names the step.
