export function cosineSimilarity(
  descriptor1: number[] | Float32Array,
  descriptor2: number[] | Float32Array,
) {
  if (descriptor1.length !== descriptor2.length) {
    throw new Error("Descriptors must have the same length");
  }

  let dotProduct = 0.0;
  let norm1 = 0.0;
  let norm2 = 0.0;

  for (let i = 0; i < descriptor1.length; i++) {
    const value1 = descriptor1[i] ?? 0;
    const value2 = descriptor2[i] ?? 0;
    dotProduct += value1 * value2;
    norm1 += Math.pow(value1, 2);
    norm2 += Math.pow(value2, 2);
  }

  const magnitude = Math.sqrt(norm1) * Math.sqrt(norm2);

  if (magnitude === 0) {
    return 0;
  }

  return dotProduct / magnitude;
}
