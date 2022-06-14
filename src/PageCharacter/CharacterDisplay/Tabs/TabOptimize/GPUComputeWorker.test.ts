import { GPU } from "gpu.js"
import { customRead, frac, greaterEq, max, min, prod, sum } from "../../../../Formula/utils"
import { precompute } from "./GPUComputeWorker"

describe("GPUComputeWorker", () => {
  describe("precompute", () => {
    const gpu = new GPU({ mode: "webgl" })
    it("can compute `sum`", () => {
      const node = sum(3, customRead(["test"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0], [[1], [2], [3], [4], [5]], [[1], [2], [3], [4], [5]])
      for (let i = 0; i < 5; i++)  for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toEqual(3 + i + j + 2)
    })
    test("can compute `prod`", () => {
      const node = prod(3, customRead(["test"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0], [[1], [2], [3], [4], [5]], [[1], [2], [3], [4], [5]])
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toEqual(3 * (i + 1 + j + 1))
    })
    test("can compute `min`", () => {
      const node = min(3, customRead(["test"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0], [[1], [2], [3], [4], [5]], [[1], [2], [3], [4], [5]])
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toEqual(Math.min(3, (i + 1 + j + 1)))
    })
    test("can compute `max`", () => {
      const node = max(3, customRead(["test"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0], [[1], [2], [3], [4], [5]], [[1], [2], [3], [4], [5]])
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toEqual(Math.max(3, (i + 1 + j + 1)))
    })
    test("can compute `sum_frac`", () => {
      const node = frac(3, customRead(["test"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0], [[1], [2], [3], [4], [5]], [[1], [2], [3], [4], [5]])
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toApproximate(3 / (3 + i + 1 + j + 1))
    })
    test("can compute `threshold`", () => {
      const node = greaterEq(customRead(["test"]), customRead(["DD"]), 24)
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      kernel.setOutput([5, 5])
      const results = kernel([0, 0], [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++)
        expect(results[i][j][0]).toEqual((i <= j) ? 24 : 0)
    })
    test("can use appropriate window", () => {
      const node = sum(customRead(["1"]), customRead(["2"]))
      const { kernel } = precompute(gpu, [node], [], [0], x => x.path[0])
      {
        kernel.setOutput([1, 5])
        const results = kernel([0, 0], [[3, 0]], [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]])
        for (let j = 0; j < 5; j++)
          expect(results[j][0][0]).toEqual(3 + j + 1)
      }
      {
        kernel.setOutput([5, 1])
        const results = kernel([0, 0], [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], [[3, 0]])
        for (let j = 0; j < 5; j++)
          expect(results[0][j][0]).toEqual(3 + j + 1)
      }
    })
    gpu.destroy()
  })
})
