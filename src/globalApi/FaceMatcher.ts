import { FaceMatch } from "../classes/FaceMatch";
import { LabeledFaceDescriptors } from "../classes/LabeledFaceDescriptors";
import { cosineSimilarity } from "../cosineSimilarity";
import { WithFaceDescriptor } from "../factories";

export class FaceMatcher {
  private _labeledDescriptors: LabeledFaceDescriptors[];
  private _similarityThreshold: number;

  constructor(
    inputs:
      | LabeledFaceDescriptors
      | WithFaceDescriptor<any>
      | Float32Array
      | Array<LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array>,
    similarityThreshold: number = 0.6,
  ) {
    this._similarityThreshold = similarityThreshold;

    const inputArray = Array.isArray(inputs) ? inputs : [inputs];

    if (!inputArray.length) {
      throw new Error(
        `FaceRecognizer.constructor - expected atleast one input`,
      );
    }

    let count = 1;
    const createUniqueLabel = () => `person ${count++}`;

    this._labeledDescriptors = inputArray.map((desc) => {
      if (desc instanceof LabeledFaceDescriptors) {
        return desc;
      }

      if (desc instanceof Float32Array) {
        return new LabeledFaceDescriptors(createUniqueLabel(), [desc]);
      }

      if (desc.descriptor && desc.descriptor instanceof Float32Array) {
        return new LabeledFaceDescriptors(createUniqueLabel(), [
          desc.descriptor,
        ]);
      }

      throw new Error(
        `FaceRecognizer.constructor - expected inputs to be of type LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array | Array<LabeledFaceDescriptors | WithFaceDescriptor<any> | Float32Array>`,
      );
    });
  }

  public get labeledDescriptors(): LabeledFaceDescriptors[] {
    return this._labeledDescriptors;
  }
  public get similarityThreshold(): number {
    return this._similarityThreshold;
  }

  public computeMeanSimilarity(
    queryDescriptor: Float32Array,
    descriptors: Float32Array[],
  ): number {
    return (
      descriptors
        .map((d) => cosineSimilarity(d, queryDescriptor))
        .reduce((s1, s2) => s1 + s2, 0) / (descriptors.length || 1)
    );
  }

  public matchDescriptor(queryDescriptor: Float32Array): FaceMatch {
    return this.labeledDescriptors
      .map(
        ({ descriptors, label }) =>
          new FaceMatch(
            label,
            this.computeMeanSimilarity(queryDescriptor, descriptors),
          ),
      )
      .reduce((best, curr) =>
        best.similarity > curr.similarity ? best : curr,
      );
  }

  public findBestMatch(queryDescriptor: Float32Array): FaceMatch {
    const bestMatch = this.matchDescriptor(queryDescriptor);
    return bestMatch.similarity >= this.similarityThreshold
      ? bestMatch
      : new FaceMatch("unknown", bestMatch.similarity);
  }

  public toJSON(): any {
    return {
      similarityThreshold: this.similarityThreshold,
      labeledDescriptors: this.labeledDescriptors.map((ld) => ld.toJSON()),
    };
  }

  public static fromJSON(json: any): FaceMatcher {
    const labeledDescriptors = json.labeledDescriptors.map((ld: any) =>
      LabeledFaceDescriptors.fromJSON(ld),
    );
    return new FaceMatcher(labeledDescriptors, json.similarityThreshold);
  }
}
