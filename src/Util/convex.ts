import { det, inv } from "mathjs"

/** `sum {coeff} * prod_{terms} x[{term.key}] ^ {term.value}` */
export type Posynomial = { coeff: number, terms: Dict<string, number> }[]

/** minimize `posy0` s.t. `posyi <= 1` */
export function minimizePosy(x: Dict<string, number>, posys: Posynomial[]): number[] {
  /** `log sum prod exp { value * y[key]) }` where `y[-1] = 1` */
  type LogPosy = { id: number, pow: number }[][]

  const keys = [...new Set(posys.flatMap(posy => posy.flatMap(mono => Object.keys(mono.terms))))]
  const keyMap = new Map(keys.map((k, i) => [k, i]))
  const y = keys.map(_ => NaN), logPosys: LogPosy[] = posys.map(posy =>
    posy.map(mono => [{ id: -1, pow: Math.log(mono.coeff) }, ...Object.entries(mono.terms).map(([id, pow]) => ({ id: keyMap.get(id)!, pow }))].filter(x => x.pow)))
  keys.forEach((k, i) => y[i] = Math.log(x[k]!))

  if (process.env.NODE_ENV === "development" &&
    logPosys.some(posy => posy.some(mono => mono.some(({ pow }) => !isFinite(pow))))) throw new Error("Ill-formed Posynomial")

  const computeMono = (y: number[], mono: LogPosy[number]) => Math.exp(mono.reduce((accu, { id, pow }) => accu + (y[id] ?? 1) * pow, 0))
  const f = (y: number[]): number[] => logPosys.map(posy => Math.log(posy.reduce((accu, mono) => accu + computeMono(y, mono), 0)))
  const dfddf = (y: number[]): { df: number[], ddf: number[][] }[] => {
    return logPosys.map(posy => {
      const monoVal = posy.map(mono => computeMono(y, mono)), sum = monoVal.reduce((a, b) => a + b, 0)
      const d = y.map(_ => 0)
      monoVal.forEach((val, i) => {
        for (const { id, pow } of posy[i])
          if (id !== -1) d[id] += pow * val
      })
      d.forEach((_, i) => d[i] /= sum)

      const dd = y.map(_ => y.map(_ => 0))
      posy.forEach((mono, im) => {
        const val = monoVal[im]
        for (const { id: id1, pow: pow1 } of mono)
          if (id1 !== -1) for (const { id: id2, pow: pow2 } of mono)
            if (id2 !== -1) dd[id1][id2] += pow1 * pow2 * val
      })
      dd.forEach((row, i) => row.forEach((e, j) => dd[i][j] = e / sum - d[i] * d[j]))
      return { df: d, ddf: dd }
    })
  }

  boundedMinimize(y, f, dfddf)
  keys.forEach((k, i) => x[k] = Math.exp(y[i]))
  return f(y).map(Math.exp)
}
/** minimize `f0` s.t. `fi <= 0` for `i > 0` */
export function boundedMinimize(x: number[], f: (x: number[]) => number[], dfddf: (x: number[]) => { df: number[], ddf: number[][] }[]): void {
  if (process.env.NODE_ENV === "development" && f(x).some((x, i) => x >= 1 && i >= 0)) throw new Error("Infeasible initial point")

  let t = 1
  const bf = (x: number[]) => {
    const [f0, ...fi] = f(x)
    return fi.reduce((accu, fi) => accu - Math.log(-fi), t * f0)
  }
  const dbfddbf = (x: number[]) => {
    const [_, ...fi] = f(x), dfddfx = dfddf(x), [df0, ...dfi] = dfddfx.map(x => x.df), [ddf0, ...ddfi] = dfddfx.map(x => x.ddf)
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
  do {
    minimize(x, bf, dbfddbf)
  } while (m / (t *= tscale) >= threshold)
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
