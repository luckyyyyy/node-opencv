declare module 'node-opencv' {
  // 模板匹配方法常量
  export const TM_CCOEFF_NORMED: number;
  export const TM_CCORR_NORMED: number;
  export const TM_SQDIFF_NORMED: number;
  export const TM_CCOEFF: number;
  export const TM_CCORR: number;
  export const TM_SQDIFF: number;

  // 图像读取模式常量
  export const IMREAD_COLOR: number;
  export const IMREAD_GRAYSCALE: number;
  export const IMREAD_UNCHANGED: number;
  export const IMREAD_ANYDEPTH: number;
  export const IMREAD_ANYCOLOR: number;

  interface Point {
      x: number;
      y: number;
  }

  interface MinMaxLocResult {
      minVal: number;
      maxVal: number;
      minLoc: Point;
      maxLoc: Point;
  }

  export type Mat = {
      rows: number;
      cols: number;
      data: Buffer;

      matchTemplateAsync(template: Mat, method: number): Promise<Mat>;
      minMaxLocAsync(): Promise<MinMaxLocResult>;
  }

  export function imdecodeAsync(buffer: Buffer, flag?: number): Promise<Mat>;
  export function imreadAsync(filename: string, flag?: number): Promise<Mat>;
}

export = module;
