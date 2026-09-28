import * as tf from "@tensorflow/tfjs-core";

import { cosineSimilarity, cosineSimilarityBatch } from "../../src";

describe("cosineSimilarity", () => {
  it("should be 1.0 for identical vectors", () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBe(1);
  });

  it("should be 1.0 for vectors with the same direction and different magnitude", () => {
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBe(1);
  });

  it("should be 0.0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBe(0);
  });

  it("should be -1.0 for opposite vectors", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBe(-1);
  });

  it("should be 0.0 if any descriptor has magnitude 0", () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
    expect(cosineSimilarity([1, 2, 3], [0, 0, 0])).toBe(0);
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });

  it("should throw an error if the descriptors have different lengths", () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrowError();
  });

  it("should work with Float32Array", () => {
    expect(
      cosineSimilarity(
        new Float32Array([1, 2, 3]),
        new Float32Array([2, 4, 6]),
      ),
    ).toBe(1);
  });

  it("should compute the expected cosine similarity", () => {
    // dot product 32, magnitude sqrt(14) * sqrt(77)
    const similarity = cosineSimilarity([1, 2, 3], [4, 5, 6]);
    expect(similarity).toBeCloseTo(0.9746, 2);
  });
});

describe("cosineSimilarityBatch", () => {
  // 128 entries, the dimension the face recognition model outputs
  const D = 128;

  const buildDescriptor = (fn: (i: number) => number) =>
    Array.from({ length: D }, (_, i) => fn(i));

  it("should return an empty array for an empty batch", async () => {
    expect(await cosineSimilarityBatch([1, 2, 3], [])).toEqual([]);
  });

  it("should throw an error if a descriptor has a different length than the query", async () => {
    const query = new Array(D).fill(1);

    await expectAsync(
      cosineSimilarityBatch(query, [new Array(D).fill(1), [1, 2, 3]]),
    ).toBeRejectedWithError(/descriptors\[1\]\.length \(3\)/);
  });

  it("should be 1.0 for a single identical descriptor", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));

    expect(await cosineSimilarityBatch(descriptor, [descriptor])).toEqual([1]);
  });

  it("should be scale invariant, like cosineSimilarity", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));
    const scaled = descriptor.map((val) => val * 1.9);

    const [batchSimilarity] = await cosineSimilarityBatch(descriptor, [scaled]);

    expect(batchSimilarity).toBeCloseTo(1, 6);
    expect(batchSimilarity).toBeCloseTo(
      cosineSimilarity(descriptor, scaled),
      6,
    );
  });

  it("should preserve the order of the input descriptors", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));
    const other = buildDescriptor((i) => Math.cos(i * 1.7));

    const result = await cosineSimilarityBatch(descriptor, [
      other,
      descriptor,
      descriptor,
    ]);

    expect(result[0]).toBeCloseTo(cosineSimilarity(descriptor, other), 6);
    expect(result[1]).toBeCloseTo(1, 6);
    expect(result[2]).toBeCloseTo(1, 6);
  });

  it("should match cosineSimilarity for every pair of a larger batch", async () => {
    const query = buildDescriptor((i) => Math.sin(i * 0.3));
    const descriptors = Array.from({ length: 50 }, (_, n) =>
      buildDescriptor((i) => Math.cos(i * (n + 1) * 0.11)),
    );

    const result = await cosineSimilarityBatch(query, descriptors);

    expect(result.length).toBe(50);
    result.forEach((similarity, i) => {
      expect(similarity).toBeCloseTo(
        cosineSimilarity(query, descriptors[i]),
        5,
      );
    });
  });

  it("should work with Float32Array input", async () => {
    const descriptor = new Float32Array(buildDescriptor((i) => Math.sin(i)));
    const other = new Float32Array(buildDescriptor((i) => Math.cos(i)));

    const result = await cosineSimilarityBatch(descriptor, [other, descriptor]);

    expect(result[0]).toBeCloseTo(cosineSimilarity(descriptor, other), 6);
    expect(result[1]).toBeCloseTo(1, 6);
  });

  it("should return 0.0 for descriptors with magnitude 0, matching cosineSimilarity", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));
    const zero = new Array(D).fill(0);

    const result = await cosineSimilarityBatch(descriptor, [zero, descriptor]);

    expect(result[0]).toBe(0);
    expect(result[0]).toBe(cosineSimilarity(descriptor, zero));
    expect(result[1]).toBeCloseTo(1, 6);
  });

  it("should return all 0.0 if the query has magnitude 0", async () => {
    const query = new Array(D).fill(0);
    const descriptors = [
      buildDescriptor((i) => Math.sin(i)),
      buildDescriptor((i) => Math.cos(i)),
    ];

    const result = await cosineSimilarityBatch(query, descriptors);

    result.forEach((similarity) => {
      expect(similarity).toBe(0);
      expect(similarity).toBe(cosineSimilarity(query, descriptors[0]));
    });
  });

  it("should never return NaN or Infinity", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));
    const result = await cosineSimilarityBatch(descriptor, [
      new Array(D).fill(0),
      descriptor,
    ]);

    result.forEach((similarity) => {
      expect(isFinite(similarity)).toBe(true);
    });
  });

  it("should not leak tensors", async () => {
    const descriptor = buildDescriptor((i) => Math.sin(i));
    const descriptors = Array.from({ length: 10 }, (_, n) =>
      buildDescriptor((i) => Math.cos(i * (n + 1) * 0.11)),
    );

    const numTensorsBefore = tf.memory().numTensors;

    await cosineSimilarityBatch(descriptor, descriptors);
    await cosineSimilarityBatch(descriptor, descriptors);

    expect(tf.memory().numTensors).toBe(numTensorsBefore);
  });
});
