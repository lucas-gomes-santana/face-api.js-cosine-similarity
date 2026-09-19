import { cosineSimilarity } from "../../../src/cosineSimilarity";

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
