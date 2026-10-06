func shuffledLoRATrainingSamples<S: Sequence, R: RandomNumberGenerator>(
  _ samples: S, using generator: inout R
) -> [S.Element] {
  var samples = Array(samples)
  samples.shuffle(using: &generator)
  return samples
}
