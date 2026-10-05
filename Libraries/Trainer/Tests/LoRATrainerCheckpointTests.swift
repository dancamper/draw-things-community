import Diffusion
import Foundation
import NNC
import Trainer
import XCTest

final class LoRATrainerCheckpointTests: XCTestCase {
  func testKrea2TextFusionAdapterRequiresMaterializedPackedConditioning() {
    let graph = DynamicGraph()
    let paddedTextLength = 64
    let promptPrefixLength = 34
    let tokenLength = 8
    let featureLength = 2_560 * 12
    let conditioning = graph.variable(
      .GPU(0), .HWC(1, paddedTextLength, featureLength), of: Float16.self)
    let slice = conditioning[
      0..<1, promptPrefixLength..<(promptPrefixLength + tokenLength), 0..<featureLength]

    XCTAssertTrue(slice.rawValue.isTensorView)
    XCTAssertTrue(slice.isContiguous)
    let contiguous = slice.contiguous()
    XCTAssertTrue(contiguous.rawValue.isTensorView)

    let materialized = contiguous.copied()
    XCTAssertFalse(materialized.rawValue.isTensorView)
    let adapter = Krea2TextFusionAdapter(
      batchSize: 1, textLength: (0, tokenLength), usesFlashAttention: .none
    ).1
    adapter.compile(inputs: materialized)
    let _ = adapter(inputs: materialized)
  }

  func testKrea2SessionSavesLoRAWeightsAndResumeStep() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let graph = DynamicGraph()
    // Session serialization only needs a compiled LoRA model, not the full Krea 2 weights.
    let model = LoRADense(
      count: 4,
      configuration: LoRANetworkConfiguration(
        rank: 2, scale: 1, highPrecision: true, testing: false),
      noBias: true, index: 27, name: "to_v")
    let input = graph.variable(.CPU, .NC(1, 4), of: Float.self)
    input.full(1)
    let _ = model(inputs: input)
    let down = try XCTUnwrap(model.parameters.first(where: { $0.contains("lora_down") }))
    let up = try XCTUnwrap(model.parameters.first(where: { $0.contains("lora_up") }))
    down.copy(from: Tensor<Float>([Float](repeating: 0.25, count: 8), .CPU, .NC(2, 4)))
    up.copy(from: Tensor<Float>([Float](repeating: 0.5, count: 8), .CPU, .NC(4, 2)))

    let path = directory.appendingPathComponent("session.ckpt").path
    LoRATrainerCheckpoint(version: .krea2, unet: model, step: 100).write(to: path)

    try graph.openStore(path, flags: .readOnly) { store in
      let step = Tensor<Int32>(from: try XCTUnwrap(store.read("current_step")))
      XCTAssertEqual(step[0], 100)
      // Resume reads the original parameter names under the same "dit" prefix as the base model.
      let downKey = "__dit__[\(down.name)]"
      let upKey = "__dit__[\(up.name)]"
      let keys = Set(store.keys.filter { $0.hasPrefix("__dit__[") })
      XCTAssertEqual(keys, Set([downKey, upKey]))
      for (key, shape, value) in [(downKey, [2, 4], Float(0.25)), (upKey, [4, 2], Float(0.5))] {
        let tensor = Tensor<Float>(from: try XCTUnwrap(store.read(key)))
        XCTAssertEqual(Array(tensor.shape), shape)
        for row in 0..<shape[0] {
          for column in 0..<shape[1] {
            XCTAssertEqual(tensor[row, column], value)
          }
        }
      }
    }.get()
  }
}
