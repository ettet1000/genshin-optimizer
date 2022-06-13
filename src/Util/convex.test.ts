import { boundedMinimize, minimize, minimizePosy, Posynomial } from "./convex"

describe("convex API", () => {
  describe("minimize", () => {
    it("can minimize functions with correct gap", () => {
      const f = ([x, y]: number[]) => Math.pow(x - 1, 4) + Math.pow(y - 2, 4)
      const x = [50, 30]
      minimize(x, f,
        ([x, y]) => ({
          df: [4 * Math.pow(x - 1, 3), 4 * Math.pow(y - 2, 3)],
          ddf: [[12 * Math.pow(x - 1, 2), 0], [0, 12 * Math.pow(y - 2, 2)],],
        }))
      expect(f(x)).toBeLessThan(1e-5)
    })
  })
  describe("boundedMinimize", () => {
    it("can minimize functions with correct gap and bound", () => {
      const x = [-4, -2], f = ([x, y]: number[]) => [
        Math.pow(x - 1, 4) + Math.pow(y - 2, 4),
        Math.pow(x + 3, 2) + Math.pow(y + 1, 2) - 4 // Circle of radius 2 around (-4, -2)
      ] // The optimal solution is ~42.3355 around (-1.296, 0.047)
      boundedMinimize(x, f, ([x, y]) => ([{
        df: [4 * Math.pow(x - 1, 3), 4 * Math.pow(y - 2, 3)],
        ddf: [[12 * Math.pow(x - 1, 2), 0], [0, 12 * Math.pow(y - 2, 2)]],
      }, {
        df: [2 * (x + 3), 2 * (y + 1)],
        ddf: [[2, 0], [0, 2]],
      }]))
      const [f0, ...fi] = f(x)
      expect(f0).toBeLessThan(43.6752)
      expect(fi[0]).toBeLessThan(0)
    })
  })
  describe("minimizePosy", () => {
    it("can minimize functions with correct gap and bound", () => {
      const v = { x: 1, y: Math.E }, posys: Posynomial[] = [
        [{ coeff: 1, terms: { x: 2 } }, { coeff: 1, terms: { y: 2 } }], // x^2 + y^2
        [{ coeff: 1, terms: { x: -1, y: -1 } }], // xy > 1
      ]
      const [f0, f1] = minimizePosy(v, posys), { x, y } = v
      expect(f0).toApproximate(2)
      expect(f1).toBeLessThan(1)
      expect(x * x + y * y).toApproximate(f0)
      expect(x * y).toApproximate(f1)
    })
  })
})
