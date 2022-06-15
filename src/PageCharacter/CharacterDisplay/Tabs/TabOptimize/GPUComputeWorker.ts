import { GPU, IKernelRunShortcut } from "gpu.js";
import { forEachNodes, mapFormulas } from "../../../../Formula/internal";
import { NumNode, ReadNode } from "../../../../Formula/type";
import { assertUnreachable, objectMap } from "../../../../Util/Util";
import type { InterimResult, Setup } from './BackgroundWorker';
import { ArtifactBuildData, ArtifactsBySlot, Build, filterArts, mergePlot, PlotData, RequestFilter } from './common';

export class GPUComputeWorker {
  builds: Build[] = []
  buildValues: number[] | undefined = undefined
  plotData: PlotData | undefined
  threshold: number = -Infinity
  maxBuilds: number

  arts: ArtifactsBySlot<{ key: number, value: number, cache: number }[]>
  kernel: IKernelRunShortcut
  initialValues: number[]

  callback: (interim: InterimResult) => void

  constructor({ arts, optimizationTarget, filters, plotBase, maxBuilds }: Setup, gpu: GPU, callback: (interim: InterimResult) => void) {
    this.maxBuilds = maxBuilds
    this.callback = callback
    const formulas = filters.map(x => x.value)
    const optIdx = formulas.length
    formulas.push(optimizationTarget)
    if (plotBase) {
      this.plotData = {}
      formulas.push(plotBase)
    }
    const { kernel, mapping } = precompute(gpu, formulas, filters.map(x => x.min), [optIdx, optIdx + 1], x => x.path[1])
    this.kernel = kernel
    this.arts = {
      base: arts.base,
      values: objectMap(arts.values, values => values.map(({ id, set, values }) =>
        ({ id, set, values: [...mapping].map(([name, key]) => ({ key, value: values[name], cache: NaN })).filter(x => x.value) })
      ))
    }
    this.initialValues = Array(mapping.size).fill(0)
    mapping.forEach((value, key) => this.initialValues[value] = arts.base[key])
  }

  compute(newThreshold: number, filter: RequestFilter) {
    if (this.threshold > newThreshold) this.threshold = newThreshold
    const { interimReport, initialValues, kernel } = this, self = this // `this` in nested functions means different things
    const preArts = filterArts(this.arts, filter)
    const arts = Object.values(preArts.values).sort((a, b) => a.length - b.length)
    const k = arts.splice(0, 2).map(arts => artsToKID(arts, initialValues.length))
    outer: for (let c = 0; c < 2; c++) {
      while (arts.length) {
        const l = arts[arts.length - 1].length
        if (l * k[c].k.length > 2500) break
        if (l * k.reduce((accu, { k }) => accu * k.length, 1) > 5e6) break outer
        k[c] = cartesianArts(k[c], artsToKID(arts.pop()!, initialValues.length))
      }
    }
    arts.reverse()
    const [k1, k2] = k
    this.kernel.setOutput(k.map(x => x.k.length))

    const ids: string[] = Array(arts.length).fill(""), buffer = [...initialValues]
    let count = { tested: 0, failed: 0, skipped: 0 }

    function permute(i: number) {
      if (i < 0) {
        const results = kernel(buffer, k1.k, k2.k), { threshold, builds, plotData } = self

        k1.id.forEach((id1, i) => {
          k2.id.forEach((id2, j) => {
            const result = results[j][i], value = result[0]
            if (value === -Infinity) {
              count.failed++
              return
            }
            let build: typeof builds[number] | undefined
            if (value >= threshold) {
              build = { value, artifactIds: [...id1, ...id2, ...ids] }
              builds.push(build)
            }

            if (plotData) {
              const x = result[1]
              if (!plotData[x] || plotData[x]!.value < value) {
                if (!build) build = { value, artifactIds: [...id1, ...id2, ...ids] }
                build.plot = x
                plotData[x] = build
              }
            }
          })
        })
        count.tested += k1.k.length * k2.k.length
        if (count.tested > 40_000)
          interimReport(count)
        return
      }
      arts[i].forEach(art => {
        ids[i] = art.id

        for (const current of art.values) {
          const { key, value } = current
          current.cache = buffer[key]
          buffer[key] += value
        }

        permute(i - 1)

        for (const { key, cache } of art.values)
          buffer[key] = cache
      })
    }

    permute(arts.length - 1)
    this.interimReport(count)
  }

  refresh(force: boolean): void {
    const { maxBuilds } = this
    if (Object.keys(this.plotData ?? {}).length >= 100000)
      this.plotData = mergePlot([this.plotData!])

    if (this.builds.length >= 100000 || force) {
      this.builds = this.builds
        .sort((a, b) => b.value - a.value)
        .slice(0, maxBuilds)
      this.buildValues = this.builds.map(x => x.value)
      this.threshold = Math.max(this.threshold, this.buildValues[maxBuilds - 1] ?? -Infinity)
    }
  }
  interimReport = (count: { tested: number, failed: number, skipped: number }) => {
    this.refresh(false)
    this.callback({ command: "interim", buildValues: this.buildValues, ...count })
    this.buildValues = undefined
    count.tested = 0
    count.failed = 0
    count.skipped = 0
  }
}

export function precompute(gpu: GPU, formulas: NumNode[], minimum: number[], resultIdx: [number, number?], binding: (readNode: ReadNode<number>) => string) {
  formulas = mapFormulas(formulas, f => {
    const { operation } = f
    switch (operation) {
      case "add": case "mul": case "min": case "max":
        if (f.operands.length <= 2) break
        const operands = [...f.operands]
        while (operands.length >= 2) {
          const a = operands.pop()!, b = operands.pop()!
          operands.push({ operation, operands: [a, b] })
        }
        return operands[0]
    }
    return f
  }, f => f)

  const lastUses = new Map<NumNode, number>(), ids = new Map<NumNode, number>()
  const expiring = new Map<number, number>(), availableID = new Set<number>()
  const topLeveIndex = new Map(formulas.map((f, i) => [f, i]))
  const readID = new Map<string, number>()
  const commands: number[][] = []

  let nextStep = 0, nextID = 0
  forEachNodes(formulas, _ => { }, f => {
    const { operation } = f
    switch (operation) {
      case "read":
        if (f.type !== "number" || (f.accu && f.accu !== "add"))
          throw new Error(`Unsupported ${operation} node in precompute`)
        const name = binding(f)
        readID.set(name, readID.get(name) ?? readID.size)
        break
      case "add": case "min": case "max": case "mul":
      case "threshold": case "res": case "sum_frac":
        const step = nextStep++
        lastUses.set(f, nextStep)
        f.operands.forEach(op => lastUses.set(op, step))
        break
      case "const":
        if (typeof f.value !== "number")
          throw new Error("Found string constant while precomputing")
        break
      case "match": case "lookup": case "subscript":
      case "prio": case "small":
      case "data": throw new Error(`Unsupported ${operation} node in precompute`)
      default: assertUnreachable(operation)
    }
  })
  nextStep = 0
  forEachNodes(formulas, _ => { }, f => {
    const { operation } = f
    switch (operation) {
      case "add": case "min": case "max": case "mul":
      case "threshold": case "res": case "sum_frac":
        const step = nextStep++
        expiring.forEach((exp, id) => {
          if (exp === step) {
            expiring.delete(id)
            availableID.add(id)
          }
        })
        const id = (availableID.values().next().value as number | undefined) ?? (nextID++)
        availableID.delete(id)
        ids.set(f, id)
        expiring.set(id, lastUses.get(f)!)
        let commandType = NaN
        switch (operation) {
          case "add": commandType = 0; break
          case "mul": commandType = 1; break
          case "min": commandType = 2; break
          case "max": commandType = 3; break
          case "res": commandType = 4; break
          case "sum_frac": commandType = 5; break
          case "threshold": commandType = 6; break
        }

        const command: number[] = []
        for (let i = 0; i < 4; i++) {
          const op = f.operands[i]
          if (!op) {
            command.push(0, 0)
            continue
          }
          const { operation } = op
          switch (operation) {
            case "read": command.push(1, readID.get(binding(op))!); break
            case "const": command.push(2, op.value); break
            default: command.push(3, ids.get(op)!); break
          }
        }
        command.push(commandType, id)
        commands.push(command)

        const topID = topLeveIndex.get(f)
        const cutoff = minimum[topID!]
        if (cutoff !== undefined && cutoff > -Infinity)
          commands.push([3, id, 2, cutoff, 0, 0, 0, 0, 7, id])
        resultIdx.forEach((idx, i) => {
          if (formulas[idx!] === f)
            commands.push([3, id, 2, i, 0, 0, 0, 0, 8, id])
        })
    }
  })

  if (nextID > 4)
    throw new Error("Too many ids")

  const kernel = gpu.createKernel(function (i0: number[], i1: number[][], i2: number[][]) {
    const interim = [0, 0, 0, 0], finalResults = [0, 0]
    for (let i = 0; i < this.constants.size; i++) {
      const offset = 10 * i, args = [0, 0, 0, 0], commandType = this.constants.cc[offset + 8], iOut = this.constants.cc[offset + 9]
      for (let i = 0; i < 4; i++) {
        const t = this.constants.cc[offset + i * 2], val = this.constants.cc[offset + i * 2 + 1]
        if (t === 1) args[i] = i0[val] + i1[this.thread.x][val] + i2[this.thread.y][val]
        else if (t === 2) args[i] = val
        else if (t === 3) args[i] = interim[val]
        else args[i] = 0
      }
      let result = 0
      if (commandType === 0) result = args[0] + args[1] // add
      else if (commandType === 1) result = args[0] * args[1] // mul
      else if (commandType === 2) result = Math.min(args[0], args[1]) // min
      else if (commandType === 3) result = Math.max(args[0], args[1]) // max
      else if (commandType === 4) { // res
        const res = args[0]
        if (res < 0) result = 1 - res / 2
        else if (res >= 0.75) result = 1 / (res * 4 + 1)
        else result = 1 - res
      }
      else if (commandType === 5) result = args[0] / (args[0] + args[1]) // sum_frac
      else if (commandType === 6) result = (args[0] >= args[1]) ? args[2] : args[3] // threshold
      else if (commandType === 7)
        if (args[0] < args[1]) return [-Infinity, -Infinity] // cutoff
        else result = args[0]
      else {
        finalResults[args[1]] = args[0] // output
        result = args[0]
      }
      interim[iOut] = result
    }
    return finalResults
  }).setDynamicArguments(true)
    .setDynamicOutput(true)
    .setConstants({ size: commands.length, cc: commands.flat() })
  return { kernel, mapping: readID }
}
function artsToKID(arts: ArtifactBuildData<{ key: number, value: number, cache: number }[]>[], inputCount: number): { k: number[][], id: string[][] } {
  const id = arts.map(art => [art.id]), k = arts.map(art => {
    const k: number[] = Array(inputCount).fill(0)
    art.values.forEach(({ key, value }) => k[key] = value)
    return k
  })
  return { k, id }
}
function cartesianArts(x1: { k: number[][], id: string[][] }, x2: { k: number[][], id: string[][] }): { k: number[][], id: string[][] } {
  const { k: k1, id: id1 } = x1, { k: k2, id: id2 } = x2
  const k: number[][] = [], id: string[][] = []
  for (let i = 0; i < k1.length; i++) {
    for (let j = 0; j < k2.length; j++) {
      k.push(k1[i].map((x, k) => x + k2[j][k]))
      id.push([...id1[i], ...id2[j]])
    }
  }
  return { k, id }
}
export function beautifyCommandList(commands: number[][], readID: Map<string, number>) {
  const readName = Array(readID.size).fill("")
  readID.forEach((i, k) => readName[i] = k)
  return commands.map((command) => {
    let type = ""
    switch (command[8]) {
      case 0: type = "add"; break
      case 1: type = "mul"; break
      case 2: type = "min"; break
      case 3: type = "max"; break
      case 4: type = "res"; break
      case 5: type = "sumfrac"; break
      case 6: type = "threshold"; break
      case 7: type = "cutoff"; break
      case 8: type = "return"; break
    }
    const values: string[] = []
    for (let i = 0; i < 4; i++) {
      const iT = command[i * 2], data = command[i * 2 + 1]
      switch (iT) {
        case 1: values.push(`Read(${readName[data]})`); break
        case 2: values.push(`${data}`); break
        case 3: values.push(`interim${data}`); break
        default: break
      }
    }
    return [command[9], type, ...values]
  })
}
