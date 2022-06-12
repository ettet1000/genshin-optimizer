import { det, inv } from "mathjs"

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
