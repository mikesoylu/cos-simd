# cos-simd

Cosine vector similarity sort with SIMD using rust.

## Benches

For 1000x 1536dim random vectors on M1 Air:

```
cosine_similarity_sort/simd                                                                            
                        time:   [17.008 ms 17.063 ms 17.084 ms]
cosine_similarity_sort/naive                                                                            
                        time:   [78.421 ms 78.460 ms 78.525 ms]
```
