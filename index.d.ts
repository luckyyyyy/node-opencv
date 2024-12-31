/**
 * This file is part of the William Chan.
 * @author William Chan <root@williamchan.me>
 */

/// <reference types="node" />

declare namespace cv {
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

    interface Mat {
        rows: number;
        cols: number;
        data: Buffer;
        size: {
            width: number;
            height: number;
        }
        matchTemplate(template: Mat, method: number): Mat;
        matchTemplateAsync(template: Mat, method: number): Promise<Mat>;
        minMaxLocAsync(): Promise<MinMaxLocResult>;
        release(): void;
    }

    // constants
    const TM_CCOEFF_NORMED: number;
    const TM_CCORR_NORMED: number;
    const TM_SQDIFF_NORMED: number;
    const TM_CCOEFF: number;
    const TM_CCORR: number;
    const TM_SQDIFF: number;

    const IMREAD_UNCHANGED: number;
    const IMREAD_GRAYSCALE: number;
    const IMREAD_COLOR: number;
    const IMREAD_ANYDEPTH: number;
    const IMREAD_ANYCOLOR: number;
    const IMREAD_LOAD_GDAL: number;
    const IMREAD_REDUCED_GRAYSCALE_2: number;
    const IMREAD_REDUCED_COLOR_2: number;
    const IMREAD_REDUCED_GRAYSCALE_4: number;
    const IMREAD_REDUCED_COLOR_4: number;
    const IMREAD_REDUCED_GRAYSCALE_8: number;
    const IMREAD_REDUCED_COLOR_8: number;
    const IMREAD_IGNORE_ORIENTATION: number;
    // static methods
    function imdecodeAsync(buffer: Buffer, flag?: number): Promise<Mat>;
    function imreadAsync(filename: string, flag?: number): Promise<Mat>;
    function imdecode(buffer: Buffer, flag?: number): Mat;
    function imread(filename: string, flag?: number): Mat;
}

export = cv;
