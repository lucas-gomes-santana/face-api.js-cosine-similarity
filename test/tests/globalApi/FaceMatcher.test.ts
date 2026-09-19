import { FaceMatcher } from '../../../src';
import { LabeledFaceDescriptors } from '../../../src';

describe('globalApi', () => {

  describe('FaceMatcher', () => {

    const json = '{"similarityThreshold":123.321,"labeledDescriptors":[{"label":"foo","descriptors":[[1,2,3],[4,5,6]]},{"label":"bar","descriptors":[[7,8,9],[3,2,1]]}]}';
    const st = 123.321;
    const l1 = 'foo';
    const l2 = 'bar';
    const f1 = new Float32Array([1, 2, 3]);
    const f2 = new Float32Array([4, 5, 6]);
    const f3 = new Float32Array([7, 8, 9]);
    const f4 = new Float32Array([3, 2, 1]);
    const lds = [
      new LabeledFaceDescriptors(l1, [f1, f2]),
      new LabeledFaceDescriptors(l2, [f3, f4])
    ];

    it('JSON.stringify()', () => {
      expect(JSON.stringify(new FaceMatcher(lds, st))).toBe(json);
      expect(JSON.stringify({ m: new FaceMatcher(lds, st) })).toBe(`{"m":${json}}`);
      expect(JSON.stringify([ new FaceMatcher(lds, st) ])).toBe(`[${json}]`);
    });

    it('fromJSON()', () => {
      const fm = FaceMatcher.fromJSON(JSON.parse(json));

      expect(fm.similarityThreshold).toBe(st);
      expect(fm.labeledDescriptors.length).toBe(2);
      expect(fm.labeledDescriptors[0].label).toBe(l1);
      expect(fm.labeledDescriptors[0].descriptors.length).toBe(2);
      expect(fm.labeledDescriptors[0].descriptors[0]).toEqual(f1);
      expect(fm.labeledDescriptors[0].descriptors[1]).toEqual(f2);
      expect(fm.labeledDescriptors[1].label).toBe(l2);
      expect(fm.labeledDescriptors[1].descriptors.length).toBe(2);
      expect(fm.labeledDescriptors[1].descriptors[0]).toEqual(f3);
      expect(fm.labeledDescriptors[1].descriptors[1]).toEqual(f4);
    });

    it('toJSON() => fromJSON()', () => {
      const fm = FaceMatcher.fromJSON(new FaceMatcher(lds, st).toJSON());

      expect(fm.similarityThreshold).toBe(st);
      expect(fm.labeledDescriptors.length).toBe(2);
      expect(fm.labeledDescriptors[0].label).toBe(l1);
      expect(fm.labeledDescriptors[0].descriptors.length).toBe(2);
      expect(fm.labeledDescriptors[0].descriptors[0]).toEqual(f1);
      expect(fm.labeledDescriptors[0].descriptors[1]).toEqual(f2);
      expect(fm.labeledDescriptors[1].label).toBe(l2);
      expect(fm.labeledDescriptors[1].descriptors.length).toBe(2);
      expect(fm.labeledDescriptors[1].descriptors[0]).toEqual(f3);
      expect(fm.labeledDescriptors[1].descriptors[1]).toEqual(f4);
    });

    it('computeMeanSimilarity()', () => {
      const fm = new FaceMatcher(lds, 0.6);

      expect(fm.computeMeanSimilarity(f1, [f1])).toBe(1);
      expect(fm.computeMeanSimilarity(f1, [f1, f2])).toBeCloseTo(0.9873, 2);
    });

    it('matchDescriptor() - selects the label with the highest similarity', () => {
      const fm = new FaceMatcher(lds, 0.6);

      expect(fm.matchDescriptor(f1).label).toBe(l1);
      expect(fm.matchDescriptor(f4).label).toBe(l2);
    });

    it('findBestMatch() - returns the best match above the threshold', () => {
      const fm = new FaceMatcher(lds, 0.9);

      const match = fm.findBestMatch(f1);
      expect(match.label).toBe(l1);
    });

    it('findBestMatch() - returns unknown if below the threshold', () => {
      const fm = new FaceMatcher(lds, 0.99999);

      const match = fm.findBestMatch(f4);
      expect(match.label).toBe('unknown');
    });

  });

});