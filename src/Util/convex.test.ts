import { minimize } from "./convex"

describe("convex API", () => {
  describe("minimize", () => {
    test("can minimize functions with correct gap", () => {
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
})
