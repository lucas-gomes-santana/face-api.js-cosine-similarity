import * as tf from "@tensorflow/tfjs-core";

export function cosineSimilarity(
  descriptor1: number[] | Float32Array,
  descriptor2: number[] | Float32Array,
): number {
  if (descriptor1.length !== descriptor2.length) {
    throw new Error("Descriptors must have the same length");
  }

  let dotProduct = 0.0;
  let norm1 = 0.0;
  let norm2 = 0.0;

  for (let i = 0; i < descriptor1.length; i++) {
    const value1 = descriptor1[i];
    const value2 = descriptor2[i];
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

/**
 * Computes the cosine similarity of a single query descriptor against a batch
 * of descriptors, in one matrix-vector multiplication.
 *
 * This is the vectorized counterpart of calling cosineSimilarity in a loop.
 * Because the cosine similarity is a dot product of unit vectors, normalizing
 * both sides once up front collapses the whole batch into a single matmul,
 * which runs as one native BLAS call instead of one interpreted loop per
 * descriptor. The values returned are identical to the ones cosineSimilarity
 * would produce for each pair.
 *
 * @param query The face descriptor to compare against.
 * @param descriptors The face descriptors to compare with. Every descriptor
 * must have the same length as query. May be empty.
 * @returns The cosine similarity of query against each descriptor, in the same
 * order they were passed in.
 */
export async function cosineSimilarityBatch(
  query: number[] | Float32Array,
  descriptors: Array<number[] | Float32Array>,
): Promise<number[]> {
  if (!descriptors.length) {
    return [];
  }

  const dimensions = query.length;

  for (let i = 0; i < descriptors.length; i++) {
    if (descriptors[i].length !== dimensions) {
      throw new Error(
        `cosineSimilarityBatch: descriptors[${i}].length (${descriptors[i].length}) !== query.length (${dimensions})`,
      );
    }
  }

  // tf.tidy has to be exited synchronously, so the result tensor is kept and
  // disposed once its data has been read below
  const similarities = tf.tidy(() => {
    const matrix = tf.tensor2d(descriptors as number[][], [
      descriptors.length,
      dimensions,
    ]);
    const queryTensor = tf.tensor1d(query as number[]);

    // reducing every row and the query to unit length turns the cosine
    // similarity into a plain dot product per row
    const normalizedMatrix = matrix.div(
      matrix.norm("euclidean", 1).reshape([-1, 1]),
    );
    const normalizedQuery = queryTensor.div(queryTensor.norm("euclidean"));

    const dots = normalizedMatrix.dot(normalizedQuery) as tf.Tensor1D;

    // a descriptor with a magnitude of 0 has no direction, so normalizing it
    // divides by zero and yields NaN or Inf. cosineSimilarity returns 0 for the
    // same input, so those entries are clamped to keep both APIs consistent
    return tf.where(tf.isFinite(dots), dots, tf.zerosLike(dots)) as tf.Tensor1D;
  });

  try {
    return Array.from((await similarities.data()) as Float32Array);
  } finally {
    similarities.dispose();
  }
}
