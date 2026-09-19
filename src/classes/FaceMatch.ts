import { round } from '../utils';

export interface IFaceMatch {
  label: string
  similarity: number
}

export class FaceMatch implements IFaceMatch {
  private _label: string
  private _similarity: number

  constructor(label: string, similarity: number) {
    this._label = label
    this._similarity = similarity
  }

  public get label(): string { return this._label }
  public get similarity(): number { return this._similarity }

  public toString(withSimilarity: boolean = true): string {
    return `${this.label}${withSimilarity ? ` (${round(this.similarity)})` : ''}`
  }
}