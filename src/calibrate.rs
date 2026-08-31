//! The calibrated-fusion arithmetic (spec 2026-08-30): FTS5's idf with its
//! clamp, the query's BM25 upper bound, and the logistic that fuses one
//! candidate's two lane features into a probability.
//!
//! Pure math. No store, no model, no I/O — `search::apply_calibrated_scores`
//! gathers the inputs and this module only combines them, which is what makes
//! every formula testable against hand-computed values.

/// FTS5's own k1. The bound must saturate where the scorer saturates, so this
/// is SQLite's constant and not a knob.
pub const K1: f64 = 1.2;

/// FTS5's idf: `ln((N - df + 0.5) / (df + 0.5))`, clamped the way FTS5 clamps
/// it, so a term in most of the corpus contributes almost nothing to the bound
/// instead of subtracting from it.
pub fn idf(n_rows: u64, doc_freq: u64) -> f64 {
    let n = n_rows as f64;
    let df = doc_freq as f64;
    (((n - df + 0.5) / (df + 0.5)).ln()).max(1e-6)
}

/// The query's maximum attainable BM25: each term saturates at
/// `idf · (k1 + 1)` as its frequency grows, scaled by the heaviest column
/// weight, because the best row can put every occurrence in the heaviest
/// column.
pub fn upper_bound(idfs: &[f64], max_column_weight: f64) -> f64 {
    (K1 + 1.0) * max_column_weight * idfs.iter().sum::<f64>()
}

/// The fraction of the maximal possible lexical match for this query, in
/// `[0, 1]`. A score outside the meaningful range reads as no evidence rather
/// than inverting.
pub fn normalize_bm25(score: f64, bound: f64) -> f64 {
    if bound <= 0.0 {
        return 0.0;
    }
    (score / bound).clamp(0.0, 1.0)
}

/// The fused probability: `sigma(w_s·cos + w_k·bm25n + b)`. The coefficients
/// are `[calibrated]`'s, fit on the ground-truth pool — see the spec's
/// "The rule" for what the near-equality of the two weights means.
pub fn probability(cos: f64, bm25n: f64, params: &crate::config::CalibratedConfig) -> f64 {
    let z = params.semantic * cos + params.keyword * bm25n + params.intercept;
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CalibratedConfig;

    #[test]
    fn idf_matches_the_fts5_formula() {
        // ln((100 - 10 + 0.5) / (10 + 0.5)) = ln(8.6190...) = 2.1539...
        assert!((idf(100, 10) - 2.1539).abs() < 1e-3);
    }

    #[test]
    fn idf_clamps_a_saturated_term_instead_of_going_negative() {
        // df > n/2 makes the raw formula negative; FTS5 clamps, so we clamp.
        assert_eq!(idf(10, 9), 1e-6);
        assert_eq!(idf(0, 0), 1e-6);
    }

    #[test]
    fn the_bound_is_k1_plus_one_times_the_idf_sum_scaled_by_the_top_weight() {
        assert!((upper_bound(&[2.0, 3.0], 1.0) - 11.0).abs() < 1e-9);
        assert!((upper_bound(&[2.0, 3.0], 2.0) - 22.0).abs() < 1e-9);
        assert_eq!(upper_bound(&[], 1.0), 0.0);
    }

    #[test]
    fn normalize_bm25_is_the_score_over_the_bound_clamped_to_the_unit_interval() {
        assert!((normalize_bm25(5.5, 11.0) - 0.5).abs() < 1e-9);
        assert_eq!(normalize_bm25(20.0, 11.0), 1.0, "capped at 1.0");
        assert_eq!(
            normalize_bm25(-1.0, 11.0),
            0.0,
            "a negative reads as no evidence"
        );
        assert_eq!(normalize_bm25(5.0, 0.0), 0.0, "no bound, no evidence");
    }

    #[test]
    fn probability_is_the_logistic_over_the_pin_coefficients() {
        let p = CalibratedConfig::default();
        assert!(
            probability(0.0, 0.0, &p) < 0.01,
            "no evidence is not an answer"
        );
        assert!(probability(1.0, 1.0, &p) > 0.99, "full evidence saturates");
        // sigma(-5.848) = 0.00288...
        assert!((probability(0.0, 0.0, &p) - 0.00288).abs() < 1e-4);
    }

    #[test]
    fn probability_is_monotone_in_each_feature() {
        let p = CalibratedConfig::default();
        assert!(probability(0.6, 0.2, &p) > probability(0.5, 0.2, &p));
        assert!(probability(0.5, 0.3, &p) > probability(0.5, 0.2, &p));
    }

    #[test]
    fn zeroed_coefficients_give_an_uninformed_half() {
        let p = CalibratedConfig {
            semantic: 0.0,
            keyword: 0.0,
            intercept: 0.0,
            ..CalibratedConfig::default()
        };
        assert!((probability(0.9, 0.9, &p) - 0.5).abs() < 1e-9);
    }
}
