# The exactness window

This document proves that the four predicates in `arithmetica` return the exact sign for every finite input. It covers the one part of that argument that is not in the literature: Shewchuk's proofs assume an unbounded exponent range, and binary64 does not have one. The *exactness window* is the range of input magnitudes within which binary64 provably behaves as the unbounded model does, and the lemmas below establish it.

The module documentation of `src/range.rs` summarizes this document. Throughout, the argument separates what is **proved here**, what is **cited**, and what is **established by inspecting the code**.

---

## 1. Setting and notation

- **F** is binary64: precision 53, normal range 2^-1022 to just under 2^1024, subnormals equal to the multiples of 2^-1074 below 2^-1022, round-to-nearest-even, gradual underflow. Rust never enables flush-to-zero.
- **F∞** is the same format with an unbounded exponent range. Shewchuk (1997) proves everything in this model: his §2 assumes that overflow and underflow do not occur.
- RN and RN∞ are the two rounding functions. ε = 2^-53, u_N = 2^-1022, u_S = 2^-1074.
- **d** is the degree of the predicate's determinant: `orient2d` 2, `orient3d` 3, `incircle` 4, `insphere` 5.
- For one call, over the non-zero input coordinates:
  - **E** is the largest exponent of a leading bit, so every coordinate satisfies |c| < 2^(E+1).
  - **q** is the smallest exponent of a lowest set bit, so every coordinate is an integer multiple of 2^q.
- **D** = 2^(E+2). Every coordinate difference, exact or rounded, has magnitude at most D.

The **window** for a predicate is the set of inputs with `q ≥ q_min` and `E ≤ e_max` (§7 gives the numbers).

---

## 2. Lemma Z: a floating-point difference is zero only when its operands are equal

**Statement.** For finite x, y in F: `RN(x − y) = 0` if and only if `x = y`.

**Proof.** Every finite element of F is a multiple of 2^-1074, so x − y is too. If x ≠ y then |x − y| ≥ 2^-1074. RN is monotone and 2^-1074 is in F, so |RN(x − y)| ≥ 2^-1074. If the difference overflows it is ±∞, which is not zero. ∎

**Use.** This is what makes the `orient2d` zero filter exact (Bartels et al. 2023, Definition 6). The filter returns `Zero` when `(acx == 0 || bcy == 0) && (acy == 0 || bcx == 0)`, or when `pa == pb`. In the first clause each product of the determinant has a factor whose exact value is zero, so the exact determinant is 0 − 0. In the second, two rows of the determinant are equal. `==` on finite `f64` is exact, and `-0.0 == 0.0` is correct because they are the same real number.

---

## 3. Theorem A: the raw Stage A filter is valid for every finite input

**Statement.** Let `det` be the floating-point determinant and `mag` the magnitude built by the underflow-protected rules of Bartels et al. in the same association order as `det`. With `bound = A ⊗ mag ⊕ u_S`, where A is the predicate's Stage A constant: if `det > bound` or `−det > bound`, the sign of `det` is the sign of the exact determinant.

**What is cited.** Bartels, Fisikopoulos & Weiser (2023), Theorem 1(2), with rules R3, R6,UFP and R7,UFP of Definition 5.

**What is verified here: the constants.** Applying the rules to the crate's four expression trees gives the smallest admissible constant a₄ for each. Evaluated in exact rational arithmetic:

| predicate | the rules require a₄ greater than | Shewchuk's constant (stored value) | meets it |
|---|---|---|---|
| `orient2d` | (3 + 12ε + …)ε | (3 + 16ε)ε | yes, by 4ε² |
| `orient3d` | (7 + 42ε + …)ε | (7 + 56ε)ε | yes, by 14ε² |
| `incircle` | (10 + 75ε + …)ε | (10 + 96ε)ε | yes, by 21ε² |
| `insphere` | (15 + 150ε + …)ε | (16 + 224ε)ε | yes, by about ε |

The margin is thin for `orient2d`: a constant of (3 + 8ε)ε, which is the value of Shewchuk's `resulterrbound`, would not meet the requirement. The trees used: products of two differences by R7 on two R3 leaves; sums and differences by R6; the outer sums in the association order the code uses, which is `(t1 ⊕ t2) ⊕ t3` for `orient3d` and `incircle` and `(t1 ⊖ t2) ⊕ (t3 ⊖ t4)` for `insphere`.

**What is proved here: overflow (Lemma O).** The theorem's proof works with finite intermediates. Finite inputs near `f64::MAX` can overflow a difference or a product, so the overflow case needs its own argument.

*Claim.* If any intermediate of `det` is ±∞ or NaN, the filter certifies nothing.

*Proof.* A polynomial evaluated with only ⊕, ⊖ and ⊗ cannot turn a non-finite intermediate back into a finite value: ∞ ± finite and ∞ × non-zero are infinite, and ∞ − ∞ and ∞ × 0 are NaN. So `det` itself is ±∞ or NaN. By induction over the tree, wherever a subexpression q̃ is not NaN its magnitude m satisfies m ≥ |q̃| or m is NaN, and wherever q̃ is NaN its magnitude is ∞ or NaN:

- A difference leaf has m = |q̃| exactly.
- A product has m = m₁ ⊗ m₂ ⊕ u_N with m₁ ≥ |q̃₁| and m₂ ≥ |q̃₂|. Rounding is monotone, so m ≥ |q̃| unless m₁ ⊗ m₂ is the NaN case 0 × ∞.
- A sum has m = m₁ ⊕ m₂ ≥ |q̃₁| ⊕ |q̃₂| ≥ |q̃|. Both magnitudes are non-negative, so no ∞ − ∞ arises.

So a non-finite `det` gives a `mag` that is ∞ or NaN, `bound` is ∞ or NaN, and both strict comparisons are false. ∎

When `det` is finite but `mag` overflowed, `bound` is infinite and the filter again certifies nothing. When everything is finite, Theorem 1(2) applies as stated.

---

## 4. Lemma G: inside the window, binary64 executes exactly as F∞ does

### 4.1 Two facts about rounding

**Fact 1 (coincidence).** If a real x is a multiple of 2^-1074 and |x| is below the overflow threshold, then RN(x) = RN∞(x).

*Proof.* If |x| ≥ 2^-1022, F and F∞ have the same elements around x, so the nearest element is the same. If |x| < 2^-1022, x is a subnormal of F, so RN(x) = x; it has at most 52 significant bits, so it is in F∞ too and RN∞(x) = x. ∎

**Fact 2 (rounding keeps the grid).** If x is a multiple of 2^g, so is RN∞(x).

*Proof.* If x is in F∞ there is nothing to show. Otherwise x needs more than 53 bits, so its leading bit is at 2^(g+53) or above, and RN∞(x) is a multiple of the unit in the last place there, which is at least 2^(g+1). ∎

### 4.2 The structural properties the lemma uses

These are properties of the code in the in-window path: the Shewchuk-form Stage A and the four `*_adapt` functions with the `expansion` primitives they call. They are established by inspecting the code; §8 lists the tests that check their consequences.

- **P1 (operations).** Every floating-point operation is an addition, a subtraction, a multiplication, a comparison or an absolute value. The only literal multipliers are 2.0, ±1.0 and the splitter 2^27 + 1, all integers. The only other constants are the error-bound constants, and a product with one of those feeds a comparison and nothing else.
- **P2 (homogeneity).** Every value has a degree j between 1 and d: coordinates, differences and tails have degree 1; a sum or difference combines two values of the same degree; a product has the sum of its operands' degrees.
- **P3 (size).** Every expansion the code builds is a formal sum of at most K products of differences and tails, each product of degree at most d. By inspection K is at most 8, 48, 192 and 72 for `orient2d`, `orient3d`, `incircle` and `insphere` (each difference counted as its rounded value plus its tail). The proof needs only **K ≤ 2^10**.

### 4.3 Statement

Let the coordinates of one call satisfy `d·min(q, 0) ≥ −1074`, and suppose no operation exceeds the overflow threshold (Lemma H gives the condition). Then, excluding the error-bound products, every operation of the in-window path returns the same value in F as in F∞. If also `c·2^(d·min(q,0)) ≥ 2^-1022` for the smallest error-bound constant c, the error-bound products agree too, and so every comparison and every branch agrees.

### 4.4 Proof

Run the F∞ execution and the F execution side by side, and induct on the operation count. Hypothesis: every value so far is identical in the two executions, and every value of degree j is a multiple of 2^(j·q).

- *Base.* Coordinates are multiples of 2^q by the definition of q.
- *Step, sum or difference.* Two multiples of 2^(j·q) have an exact sum that is a multiple of 2^(j·q).
- *Step, product.* A multiple of 2^(i·q) times a multiple of 2^(j·q) is a multiple of 2^((i+j)·q). A product with an integer literal stays on its grid.
- In both cases the exact result r is a multiple of 2^(j·q) with j ≤ d. If q ≥ 0 that is an integer. If q < 0 then j·q ≥ d·q ≥ −1074. Either way r is a multiple of 2^-1074, so RN(r) = RN∞(r) by Fact 1, and the rounded value is still a multiple of 2^(j·q) by Fact 2.

That covers every operation except the error-bound products. Those have the form c ⊗ v, where v is a permanent or an estimate of the determinant: a degree-d value, so either zero or at least 2^(d·min(q,0)) in magnitude. If v = 0 the product is 0 in both models. Otherwise c·v ≥ 2^-1022 by the second hypothesis, the product is in the normal range, and RN = RN∞ there. A sum of two such products is in the normal range as well. ∎

**Consequence.** Inside the window every theorem of Shewchuk (1997) holds for the binary64 execution: `two_sum` and `two_product` are exact, expansions are nonoverlapping and sum to the exact value, and the Stage A, B and C error bounds are valid with his published constants.

---

## 5. Lemma H: no operation overflows inside the window

**Statement.** In the F∞ execution of the in-window path, every value has magnitude at most
`(1 + δ) · 2^27.0001 · K · max(1, D)^d`, with δ < 2^-20.

**Proof.** Call the *formal absolute value* A(t) of an expression tree t the value it would have if every leaf were replaced by its absolute value and every subtraction by an addition.

1. *The exact operations barely increase the sum of magnitudes.* `two_sum(a, b)` returns (x, y) with x + y = a + b, |x| ≤ (|a| + |b|)(1 + ε) and |y| ≤ ε|x|, so |x| + |y| ≤ (|a| + |b|)(1 + 2^-51). `two_product(a, b)` is the same with |a||b|. `grow_expansion`, `expansion_sum` and `scale_expansion` are sequences of these. By induction, the components of any expansion built from a tree t satisfy Σ|component| ≤ A(t)·(1 + 2^-51)^N, where N is the number of operations performed. N is far below 2^30 (the largest expansion has 1,152 components), which gives δ < 2^-20.
2. *A(t) is at most K·max(1, D)^d.* Each leaf has magnitude at most D, and by P3 the tree expands to at most K products of at most d leaves.
3. *Values inside the primitives.* Inside `two_sum`, each intermediate is at most four times the sum of the operand magnitudes. Inside `two_product`, `split(a)` forms (2^27 + 1)·a, and each partial product and partial sum is at most four times |a||b|. So no intermediate exceeds 2^27.0001 times a value that steps 1 and 2 already bound. The product a·b is itself the value of a tree, so step 2 bounds it directly; there is no K² term.
4. The plain floating-point Stage A and Stage C evaluations are trees of the same shape and are bounded by step 2. ∎

The absolute-sum argument is needed. A bound of the form "each component is at most twice the expansion's value" is false for nonoverlapping expansions: 2^k − 2^(k−1) − … − 2^(k−n) is nonoverlapping, and its value is 2^(k−n).

---

## 6. Two smaller lemmas

**Lemma S (the rescale is lossless).** If an input is outside the window but `E − q ≤ e_max − q_min`, there is an integer k with `q + k ≥ q_min` and `E + k ≤ e_max`. Multiplying every coordinate by 2^k is exact, and it multiplies the determinant by 2^(d·k), which is positive.

*Proof.* Take any k between `q_min − q` and `e_max − E`; the interval is non-empty by hypothesis. Each scaled coordinate has its lowest bit at or above 2^q_min and its leading bit at or below 2^e_max, so it is a normal `f64` with the same significand and the multiplication is exact. From q ≥ −1074 and E ≤ 1023, the chosen k (the lower end when scaling up, the upper end when scaling down) satisfies |k| ≤ 894, so 2^k is a normal power of two. ∎

**Lemma E (an estimate never has the opposite sign).** Let e₁ … e_m be a nonoverlapping expansion in increasing order of magnitude with exact sum x, and let Q be the left-to-right floating-point sum of its components. Then sign(Q) is sign(x) or zero.

*Proof.* Let e_m be the largest non-zero component, with leading bit 2^t and lowest set bit 2^l. Nonoverlapping means the other components lie strictly below 2^l, so sign(x) = sign(e_m). The floating-point partial sum of the magnitudes of the lower components is at most 2^l: by induction each partial sum is at most the next power of two above its top component, which is representable, and rounding is monotone. So the partial sum Q′ before the last addition has |Q′| ≤ 2^l ≤ |e_m|. Then Q′ + e_m is zero or has the sign of e_m, and rounding preserves that. ∎

*Use.* Where a path returns the sign of an estimate of an exact expansion, a non-zero result is correct, and a zero result goes to Stage 4.

---

## 7. The window and its margins

### 7.1 The limits

| predicate | d | `q_min` | `e_max` | d·`q_min` | d·(`e_max` + 2) | widest spread `E − q` handled without Stage 4 |
|---|---:|---:|---:|---:|---:|---:|
| `orient2d` | 2 | −450 | 478 | −900 | 960 | 928 bits |
| `orient3d` | 3 | −300 | 318 | −900 | 960 | 618 bits |
| `incircle` | 4 | −225 | 238 | −900 | 960 | 463 bits |
| `insphere` | 5 | −180 | 190 | −900 | 960 | 370 bits |

The same two products, −900 and 960, are used for all four predicates; both are divisible by 2, 3, 4 and 5.

### 7.2 The small side: d·q ≥ −900

Two conditions come from Lemma G.

| condition | requirement | chosen | spare |
|---|---|---|---|
| Exact values stay on the 2^-1074 grid | d·q ≥ −1074 | −900 | 174 bits |
| The smallest error-bound product stays normal | d·q ≥ −919 | −900 | 19 bits |

The second is the binding one. The smallest constant is `CCW_ERR_BOUND_C` = (9 + 64ε)ε² ≈ 2^-102.83. The `orient3d`, `incircle` and `insphere` C constants are larger (2^-101.30, 2^-100.54, 2^-99.85), and the A and B constants and `RESULT_ERR_BOUND` are near 2^-50. A non-zero degree-d value is at least 2^(d·q), so the smallest product is at least 2^(−102.84 + d·q), and requiring 2^-1022 gives d·q ≥ −919.16. The margin that constrains the choice is therefore 19 bits, not 174.

### 7.3 The large side: d·(E + 2) ≤ 960

From Lemma H with D = 2^(E+2): the largest value is at most (1 + δ)·2^27.0001·K·2^(d·(E+2)), and it must stay below 2^1024.

| assumption on K | requirement | chosen | spare |
|---|---|---|---|
| K ≤ 2^10 (what the proof needs) | d·(E + 2) ≤ 986 | 960 | 26 bits |
| K ≤ 192 (what inspection found) | d·(E + 2) ≤ 989 | 960 | 29 bits |

The distance from 2^960 to 2^1023 is 63 bits, but that figure is before the splitting factor and K are taken out. The margin that constrains the choice is 26 bits.

### 7.4 What the window means for ordinary data

A coordinate with leading bit 2^e has its lowest set bit at 2^(e−52) or above. So any `orient2d` input whose non-zero coordinates all lie between about 2^-398 and 2^478 is inside the window as given, with no scaling. Projected coordinates in meters between 2^20 and 2^21 (about 1.05·10^6 to 2.1·10^6) have q ≥ −32. The window scan runs only after the raw Stage A has declined to certify, so it is never on the fast path.

---

## 8. Correctness of the predicates

**Theorem W.** Every predicate returns the exact sign for every finite input, given three things:

| | assumption | status |
|---|---|---|
| **S** | Shewchuk (1997)'s theorems and Stage A to C error bounds hold in F∞ | Cited: the paper's own proofs |
| **P** | The in-window code has properties P1 to P3, and returns a non-zero sign only from (a) an estimate guarded by a Shewchuk error bound, (b) a tails-zero exit that reads an exact expansion, or (c) the sign of the final exact expansion | Established by inspection of the `*_adapt` functions, which are direct ports of Shewchuk's code |
| **I** | The Stage 4 integer path is exact for every finite input | Exact integer arithmetic in fixed widths whose sufficiency is derived in `src/exact_int.rs`; checked against an independent big-integer oracle |

*Proof, by return path.*

1. Non-finite input returns `Err`.
2. The `orient2d` zero filter is exact (Lemma Z).
3. The raw Stage A is valid for every finite input (Theorem A with Lemma O).
4. An input that cannot be brought into the window goes to Stage 4 (I).
5. An input inside the window, as given or after the lossless rescale (Lemma S), executes exactly as in F∞ (Lemmas G and H). Then S and P make every non-zero result correct, with Lemma E covering the estimate-based exits.
6. A `Zero` from step 5 goes to Stage 4 on the original coordinates (I). ∎

**What this does not prove.** It does not prove P. P is a statement about a port of roughly 2,000 lines, and the argument for it is inspection plus tests, not a derivation. The tests that back it:

- The boundary tests (`tests/boundary.rs`), which place calls next to sign boundaries in five magnitude profiles and compare every result with an exact big-integer oracle. A deep run of 18.3 million such calls returns zero wrong signs and zero false zeros.
- A scale-invariance test (`tests/window_counters.rs`): inside the window, rescaling an exactly degenerate call by a power of two must not change which stage answers it. F∞ arithmetic is scale invariant, so any difference would contradict Lemma G.
- A window-edge profile: inputs with `E = e_max` and `q = q_min` exactly, which sit on both margins at once.

**The `insphere` permanent.** The in-window Stage A and Stage B for `insphere` use Shewchuk's permanent exactly as `predicates.c` computes it: the same four row terms, summed left to right in the order a, b, c, d, although the determinant itself is summed pairwise. The in-window code therefore matches his for `insphere`, as it does for `orient3d` and `incircle`, and assumption S covers its bounds as he derived them. The raw Stage A is different by design: it sums its magnitude pairwise, matching the determinant, because the rules of Bartels et al. require the magnitude to follow the determinant's association order (§3).

---

## 9. Why the filter is not scale invariant

The scale-invariance test in §8 is restricted to exactly degenerate input. The reason is a property of Theorem A, not of Lemma G: **the underflow-protected filter is deliberately not scale invariant.** Its bound is `a₄ ⊗ mag ⊕ u_S`, and `u_N` and `u_S` are absolute constants that do not shrink when the input does. So for a determinant that is small but non-zero, the filter declines at small scales and certifies at large ones. Both outcomes are correct, since a filter never returns a wrong sign and only declines to answer, but the stage that answers such a call depends on scale. An exactly degenerate call has no such ambiguity: the comparison is a strict `>` against a bound of at least `u_S > 0`, so a zero determinant is never certified at any scale.

That is the cost of a filter that is valid for every finite input. The alternative is a filter that is valid only where no underflow occurs.

---

## 10. References

- J. R. Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*, Discrete & Computational Geometry 18(3), 305–363 (1997). §2 (the unbounded-exponent assumption) and the error bounds of the adaptive predicates.
- T. Bartels, V. Fisikopoulos, M. Weiser, *Fast Floating-Point Filters for Robust Predicates*, BIT Numerical Mathematics 63:31 (2023). Definition 5, Theorem 1(2), Definition 6.
