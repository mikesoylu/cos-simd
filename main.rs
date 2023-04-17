use cos_simd::{cosine_similarity, sort_by_cosine_similarity};

fn main() {
    let ref_vec = vec![0.2, 0.5, 0.3, 0.1, 0.9];

    let mut vectors = vec![
        vec![0.2, 0.5, 0.3, 0.1, 0.9],
        vec![0.6, 0.8, 0.5, 0.3, 0.2],
        vec![-0.1, -0.9, -0.1, -0.6, -0.4],
        vec![0.5, 0.7, 0.9, 0.4, 0.1],
        vec![0.1, 0.9, 0.1, 0.6, 0.4],
    ];

    println!("Before sorting: {:?}", vectors);
    sort_by_cosine_similarity(&ref_vec, &mut vectors);
    println!("After sorting: {:?}", vectors);
}
