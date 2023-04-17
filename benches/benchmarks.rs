use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use cos_simd::{sort_by_cosine_similarity};

fn naive_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let a_magnitude = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let b_magnitude = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (a_magnitude * b_magnitude)
}

fn naive_sort_by_cosine_similarity(ref_vec: &Vec<f32>, vectors: &mut [Vec<f32>]) {
    vectors.sort_by(|a, b| {
        let a_similarity = naive_cosine_similarity(a, &ref_vec);
        let b_similarity = naive_cosine_similarity(b, &ref_vec);
        a_similarity.partial_cmp(&b_similarity).unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_sort_by_cosine_similarity(c: &mut Criterion) {
    let n = 1000;
    let dim = 1536;
    let vectors = generate_random_vectors(n, dim);
    let target = &generate_random_vectors(1, dim)[0];

    let mut group = c.benchmark_group("cosine_similarity_sort");
    group.sample_size(10);

    group.bench_function("simd", |b| {
        b.iter(|| {
            let mut v = vectors.clone();
            let t = target.clone();
            sort_by_cosine_similarity(black_box(&t), black_box(&mut v))
        })
    });

    group.bench_function("naive", |b| {
        b.iter(|| {
            let mut v = vectors.clone();
            let t = target.clone();
            naive_sort_by_cosine_similarity(black_box(&t), black_box(&mut v))
        })
    });

    group.finish();
}

criterion_group!{
  name = benches;
  config = Criterion::default().measurement_time(std::time::Duration::from_secs(100));
  targets = bench_sort_by_cosine_similarity
}
criterion_main!(benches);
