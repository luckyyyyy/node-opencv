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

        matchTemplateAsync(template: Mat, method: number): Promise<Mat>;
        minMaxLocAsync(): Promise<MinMaxLocResult>;
    }

    // 常量定义
    const TM_CCOEFF_NORMED: number;
    const TM_CCORR_NORMED: number;
    const TM_SQDIFF_NORMED: number;
    const TM_CCOEFF: number;
    const TM_CCORR: number;
    const TM_SQDIFF: number;

    const IMREAD_COLOR: number;
    const IMREAD_GRAYSCALE: number;
    const IMREAD_UNCHANGED: number;
    const IMREAD_ANYDEPTH: number;
    const IMREAD_ANYCOLOR: number;

    // 方法定义
    function imdecodeAsync(buffer: Buffer, flag?: number): Promise<Mat>;
    function imreadAsync(filename: string, flag?: number): Promise<Mat>;
}

export = cv;
