import { det, inv } from "mathjs"

/** minimize `f0` s.t. `fi <= 0` for `i > 0` */
export function boundedMinimize(x: number[], f: (x: number[]) => number[], dfddf: (x: number[]) => { df: number[][], ddf: number[][][] }): void {
  if (process.env.NODE_ENV === "development" && f(x).some((x, i) => x >= 1 && i >= 0)) throw new Error("Infeasible initial point")

  let t = 1
  const bf = (x: number[]) => {
    const [f0, ...fi] = f(x)
    return fi.reduce((accu, fi) => accu - Math.log(-fi), t * f0)
  }
  const dbfddbf = (x: number[]) => {
    const [_, ...fi] = f(x), { df: [df0, ...dfi], ddf: [ddf0, ...ddfi] } = dfddf(x)
    const dbf = df0.map(x => t * x)
    dfi.forEach((df, eqi) => {
      const f = fi[eqi]
      df.forEach((d, i) => dbf[i] -= d / f)
    })
    const ddbf = ddf0.map(x => x.map(x => t * x))
    ddfi.forEach((ddf, eqi) => {
      const f = fi[eqi], df = dfi[eqi]
      ddf.forEach((row, i) => row.forEach((e, j) => ddbf[i][j] -= e / f - df[i] * df[j] / f / f))
    })
    return { df: dbf, ddf: ddbf }
  }

  const m = f(x).length - 1, tscale = 2.5, threshold = 1e-6 / tscale
  for (; m / t >= threshold; t *= tscale) minimize(x, bf, dbfddbf)
}
export function minimize(x: number[], f: (x: number[]) => number, dfddf: (x: number[]) => { df: number[], ddf: number[][] }): void {
  const threshold = 1e-9
  for (let count = 0; count <= 300; count++) {
    const { df, ddf } = dfddf(x)
    const direction = det(ddf) >= 1e-5 ? inv(ddf).map(row => row.reduce((accu, e, j) => accu + e * df[j], 0)) : df
    const stepSize = lineSearch(x, direction, df, f), gap = direction.reduce((accu, d, i) => accu + d * df[i], 0) / 2
    if (gap <= threshold) return
    direction.forEach((d, i) => x[i] -= stepSize * d)
  }
}
function lineSearch(x: number[], direction: number[], dfx: number[], f: (x: number[]) => number): number {
  const alpha = 0.2, beta = 0.6, fx = f(x)
  let t = 1
  while (true) {
    const v1 = f(direction.map((d, i) => x[i] - t * d)), v2 = fx - alpha * t * direction.reduce((accu, d, i) => accu + d * dfx[i], 0)
    if (!isFinite(v1) || v1 > v2) t *= beta
    else break
  }
  return t
}
