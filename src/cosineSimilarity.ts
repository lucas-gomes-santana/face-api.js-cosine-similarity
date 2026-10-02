import * as tf from "@tensorflow/tfjs-core";

/**
 * Guards against dividing by zero, which would turn a descriptor without
 * direction into NaN instead of the zero vector it normalizes to.
 */
const MIN_NORM = 1e-12;

/**
 * Computes the cosine similarity of two face descriptors, which is scale
 * invariant: descriptors pointing in the same direction score 1.0 regardless
 * of their magnitude. A descriptor of magnitude 0 has no direction and is
 * reported as 0.0.
 *
 * @param descriptor1 The first face descriptor.
 * @param descriptor2 The second face descriptor. Must have the same length.
 * @returns The similarity of both descriptors, from -1.0 to 1.0.
 */
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
 * Computes the cosine similarity of one query descriptor against a batch of
 * descriptors. Normalizing both sides up front turns every comparison into a
 * dot product, so the whole batch resolves in a single matmul rather than one
 * loop per descriptor. Returns the same values cosineSimilarity would.
 *
 * @param query The face descriptor to compare against.
 * @param descriptors The face descriptors to compare with, all of the same
 * length as query.
 * @returns The similarity of query against each descriptor, in order.
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

  const similarities = tf.tidy(() => {
    const matrix = tf.tensor2d(descriptors as number[][], [
      descriptors.length,
      dimensions,
    ]);
    const queryTensor = tf.tensor1d(query as number[]);

    const rowNorms = tf.maximum(
      matrix.norm("euclidean", 1),
      MIN_NORM,
    ) as tf.Tensor1D;
    const queryNorm = tf.maximum(
      queryTensor.norm("euclidean"),
      MIN_NORM,
    ) as tf.Tensor;

    const normalizedMatrix = matrix.div(rowNorms.reshape([-1, 1]));
    const normalizedQuery = queryTensor.div(queryNorm);

    return normalizedMatrix.dot(normalizedQuery) as tf.Tensor1D;
  });

  try {
    return Array.from((await similarities.data()) as Float32Array);
  } finally {
    similarities.dispose();
  }
}
