import { cosineSimilarity } from "../src/cosineSimilarity";
import { WithFaceDescriptor } from "../src/factories/WithFaceDescriptor";
import { WithFaceDetection } from "../src/factories/WithFaceDetection";
import { WithFaceLandmarks } from "../src/factories/WithFaceLandmarks";
import { BoxAndLandmarksDeltas } from "./expectFaceDetectionsWithLandmarks";
import {
  ExpectedFullFaceDescription,
  expectPointClose,
  expectRectClose,
  sortByFaceDetection,
} from "./utils";

export type FullFaceDescriptionDeltas = BoxAndLandmarksDeltas & {
  maxDescriptorDelta: number;
};

/* Why this conversion: descriptor matching was migrated from euclidean distance
to cosine similarity, but the reference descriptors and the per-test tolerance
(maxDescriptorDelta) were recorded for the old metric. To keep the exact same
test strictness, that tolerance must be translated into a minimum cosine
similarity. For a recomputed descriptor b that is nearly identical to the
reference a, both share the same norm n; expanding the squared euclidean
distance gives d^2 = |a|^2 + |b|^2 - 2 (a . b) = 2 n^2 - 2 (a . b), hence
a . b = n^2 - d^2/2, and dividing by n^2 turns the dot product into the cosine
similarity cos = (a . b) / (|a| |b|) = 1 - d^2 / (2 n^2). Enforcing d < maxDelta
is therefore equivalent to requiring cos > 1 - maxDelta^2 / (2 n^2). The
conversion is a function instead of a constant because the tolerance is a
per-test parameter and every reference descriptor has its own norm. */

function euclideanDeltaToMinCosine(
  maxDelta: number,
  referenceNorm: number,
): number {
  return 1 - Math.pow(maxDelta, 2) / (2 * Math.pow(referenceNorm, 2));
}

export function expectFullFaceDescriptions(
  results: WithFaceDescriptor<WithFaceLandmarks<WithFaceDetection<{}>>>[],
  allExpectedFullFaceDescriptions: ExpectedFullFaceDescription[],
  expectedScores: number[],
  deltas: FullFaceDescriptionDeltas,
) {
  const expectedFullFaceDescriptions = expectedScores
    .map((score, i) => ({
      score,
      ...allExpectedFullFaceDescriptions[i],
    }))
    .filter((expected) => expected.score !== -1);

  const sortedResults = sortByFaceDetection(results);

  expectedFullFaceDescriptions.forEach((expected, i) => {
    const { detection, landmarks, descriptor } = sortedResults[i];
    expect(Math.abs(detection.score - expected.score)).toBeLessThan(
      deltas.maxScoreDelta,
    );
    expectRectClose(detection.box, expected.detection, deltas.maxBoxDelta);
    landmarks.positions.forEach((pt, j) =>
      expectPointClose(pt, expected.landmarks[j], deltas.maxLandmarksDelta),
    );

    const referenceNorm = Math.sqrt(
      expected.descriptor.reduce((sum, val) => sum + Math.pow(val, 2), 0),
    );
    expect(cosineSimilarity(descriptor, expected.descriptor)).toBeGreaterThan(
      euclideanDeltaToMinCosine(deltas.maxDescriptorDelta, referenceNorm),
    );
  });
}
