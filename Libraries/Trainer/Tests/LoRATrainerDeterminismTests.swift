import SFMT
import XCTest

@testable import Trainer

final class LoRATrainerDeterminismTests: XCTestCase {
  private let samples = Array(0..<20)

  func testSameSeedProducesSameOrderAcrossRepeatedRuns() {
    var firstRun = SFMT(seed: 42)
    var secondRun = SFMT(seed: 42)

    let firstRunEpochs = [
      shuffledLoRATrainingSamples(samples, using: &firstRun),
      shuffledLoRATrainingSamples(samples, using: &firstRun),
    ]
    let secondRunEpochs = [
      shuffledLoRATrainingSamples(samples, using: &secondRun),
      shuffledLoRATrainingSamples(samples, using: &secondRun),
    ]

    XCTAssertEqual(firstRunEpochs, secondRunEpochs)
    XCTAssertNotEqual(firstRunEpochs[0], samples)
  }

  func testDifferentSeedsCanProduceDifferentOrders() {
    var firstRun = SFMT(seed: 42)
    var secondRun = SFMT(seed: 43)

    XCTAssertNotEqual(
      shuffledLoRATrainingSamples(samples, using: &firstRun),
      shuffledLoRATrainingSamples(samples, using: &secondRun))
  }

  func testShuffleAdvancesTheTimestepRandomNumberStreamDeterministically() {
    var firstRun = SFMT(seed: 42)
    var secondRun = SFMT(seed: 42)
    var withoutShuffle = SFMT(seed: 42)

    let firstOrder = shuffledLoRATrainingSamples(samples, using: &firstRun)
    let secondOrder = shuffledLoRATrainingSamples(samples, using: &secondRun)
    let firstTimestepRandomValue = firstRun.next()
    let secondTimestepRandomValue = secondRun.next()

    XCTAssertEqual(firstOrder, secondOrder)
    XCTAssertEqual(firstTimestepRandomValue, secondTimestepRandomValue)
    XCTAssertNotEqual(firstTimestepRandomValue, withoutShuffle.next())
  }
}
