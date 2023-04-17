use packed_simd::f32x4;
use std::cmp::Ordering;
use rayon::prelude::*;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product_sum = 0.0;
    let mut a_magnitude_sum = 0.0;
    let mut b_magnitude_sum = 0.0;
    let simd_len = f32x4::lanes();

    let chunks = a.chunks_exact(simd_len).zip(b.chunks_exact(simd_len));
    for (a_chunk, b_chunk) in chunks {
        let a_simd = f32x4::from_slice_unaligned(a_chunk);
        let b_simd = f32x4::from_slice_unaligned(b_chunk);

        let dot_product = a_simd * b_simd;
        dot_product_sum += dot_product.sum();

        let a_magnitude = a_simd * a_simd;
        a_magnitude_sum += a_magnitude.sum();

        let b_magnitude = b_simd * b_simd;
        b_magnitude_sum += b_magnitude.sum();
    }

    dot_product_sum / (a_magnitude_sum.sqrt() * b_magnitude_sum.sqrt())
}

pub fn sort_by_cosine_similarity(ref_vec: &Vec<f32>, vectors: &mut [Vec<f32>]) {
    vectors.par_sort_unstable_by(|a, b| {
        let a_similarity = cosine_similarity(a, &ref_vec);
        let b_similarity = cosine_similarity(b, &ref_vec);
        a_similarity.partial_cmp(&b_similarity).unwrap_or(Ordering::Equal)
    });
}
